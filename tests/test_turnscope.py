"""Known answers: every call is built with exact speech times, so the right numbers are known."""
import csv
import json

import numpy as np
import pytest

from conftest import SR, make_call, write
from turnscope import analyse, analyse_file, CallError
from turnscope import analyze as an
from turnscope import vad
from turnscope.cli import main


def run(script, **kw):
    x = make_call(script, **kw)
    return analyse(x[:, 0], x[:, 1], SR)


def kinds(r):
    return [e.kind for e in r.events]


# ------------------------------------------------------------------ response delay
def test_response_delays_are_measured_to_within_60_ms():
    r = run([("customer", 1.0, 3.0), ("agent", 3.8, 6.0), ("customer", 7.0, 8.5), ("agent", 8.9, 11.0), ("customer", 12.0, 13.0), ("agent", 14.5, 16.0)])
    assert len(r.responses) == 3
    for got, want in zip(r.responses, [0.8, 0.4, 1.5]):
        assert got == pytest.approx(want, abs=0.06)
    assert r.pct(50) == pytest.approx(0.8, abs=0.06) and r.pct(90) == pytest.approx(1.4, abs=0.1)


def test_response_delay_does_not_depend_on_how_loud_the_call_is():
    s = [("customer", 1.0, 3.0), ("agent", 3.6, 5.0)]
    for lv in (0.0, -20.0, -30.0):
        assert run(s, level_db=lv).responses[0] == pytest.approx(0.6, abs=0.06)


def test_a_pause_inside_one_speakers_talk_is_not_a_turn():
    r = run([("customer", 1.0, 2.0), ("customer", 2.3, 3.5), ("agent", 4.0, 5.0)])      # 0.3 s pause: same utterance
    assert len(r.responses) == 1 and r.responses[0] == pytest.approx(0.5, abs=0.06)


def test_an_agent_continuing_after_a_pause_is_not_a_second_response():
    r = run([("customer", 1.0, 2.0), ("agent", 2.5, 3.5), ("agent", 4.5, 5.5)])
    assert len(r.responses) == 1


def test_customer_pauses_are_not_agent_responses():
    r = run([("agent", 1.0, 2.5), ("customer", 3.0, 4.0)])
    assert r.responses == [] and kinds(r) == ["customer_pause"] and r.events[0].value_s == pytest.approx(0.5, abs=0.06)


def test_a_gap_longer_than_10_seconds_is_dead_air_not_a_response():
    r = run([("customer", 1.0, 2.0), ("agent", 14.0, 15.0)])
    assert r.responses == [] and r.dead_air and r.dead_air[0][0] == pytest.approx(2.0, abs=0.1)


# ------------------------------------------------------------------ overlap
def test_the_agent_starting_over_the_customer_is_an_interruption():
    r = run([("customer", 1.0, 4.0), ("agent", 3.0, 6.0)])
    assert kinds(r) == ["agent_interrupts"] and r.interrupts == 1 and r.responses == []
    assert r.events[0].value_s == pytest.approx(1.0, abs=0.1) and r.events[0].at_s == pytest.approx(3.0, abs=0.06)


def test_a_customer_barge_in_where_the_agent_stops_is_not_counted_as_not_yielding():
    r = run([("agent", 1.0, 5.0), ("customer", 4.0, 7.0)])
    assert kinds(r) == ["customer_barge_in"] and r.agent_not_yielding == 0
    assert r.events[0].value_s == pytest.approx(1.0, abs=0.1) and r.events[0].detail == "agent stopped"


def test_a_barge_in_the_agent_talks_through_is_counted_as_not_yielding():
    r = run([("agent", 1.0, 8.0), ("customer", 4.0, 5.5)])
    assert r.barge_ins == 1 and r.agent_not_yielding == 1 and r.events[0].detail == "agent talked on"


def test_a_very_short_overlap_is_ordinary_conversation_and_not_counted():
    r = run([("customer", 1.0, 3.0), ("agent", 2.9, 5.0)])                      # 0.1 s overlap
    assert r.interrupts == 0 and r.responses == [] and kinds(r) == []


# ------------------------------------------------------------------ leakage between channels
@pytest.mark.parametrize("bleed", [-30.0, -20.0, -16.0])
def test_leakage_is_not_mistaken_for_the_other_person_speaking(bleed):
    r = run([("customer", 1.0, 3.0), ("agent", 3.8, 6.0), ("customer", 7.0, 8.0)], bleed_db=bleed)
    assert r.customer_speech_s == pytest.approx(3.0, abs=0.4) and r.agent_speech_s == pytest.approx(2.2, abs=0.4)
    assert r.interrupts == 0 and r.barge_ins == 0 and r.responses[0] == pytest.approx(0.8, abs=0.1)


@pytest.mark.parametrize("bleed", [-30.0, -20.0])
def test_the_leakage_level_is_estimated_within_3_db(bleed):
    r = run([("customer", 1.0, 4.0), ("agent", 5.0, 8.0), ("customer", 9.0, 11.0)], bleed_db=bleed)
    assert r.leak_agent_to_customer_db == pytest.approx(bleed, abs=3) and r.leak_customer_to_agent_db == pytest.approx(bleed, abs=3)
    assert r.warnings == []


def test_strong_leakage_is_reported_as_a_warning():
    r = run([("customer", 1.0, 4.0), ("agent", 5.0, 8.0), ("customer", 9.0, 11.0)], bleed_db=-9.0)
    assert any("strong leakage" in w for w in r.warnings)


def test_real_overlap_is_still_seen_when_there_is_leakage():
    r = run([("customer", 1.0, 5.0), ("agent", 3.0, 7.0), ("customer", 9.0, 10.0), ("agent", 11.0, 12.0)], bleed_db=-20.0)
    assert r.interrupts == 1 and r.events[0].value_s == pytest.approx(2.0, abs=0.25)


# ------------------------------------------------------------------ no speech
def test_a_silent_or_noise_only_channel_has_no_speech_and_says_so():
    x = make_call([("customer", 1.0, 3.0), ("customer", 4.0, 6.0)])
    r = analyse(x[:, 0], x[:, 1], SR)
    assert r.agent_speech_s < 0.2 and any("agent channel" in w for w in r.warnings)
    assert r.customer_speech_s == pytest.approx(4.0, abs=0.4)


def test_steady_hum_is_not_speech():
    t = np.arange(SR * 10) / SR
    hum = (0.05 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    x = make_call([("customer", 1.0, 3.0), ("customer", 5.0, 8.0)], seconds=10.0)
    r = analyse(hum, x[:, 1], SR)
    assert r.agent_speech_s < 0.2


def test_a_completely_digital_silent_channel_does_not_crash():
    x = make_call([("customer", 1.0, 3.0)])
    r = analyse(np.zeros(len(x), dtype=np.float32), x[:, 1], SR)
    assert r.agent_speech_s == 0 and r.warnings


def test_very_short_audio_does_not_crash():
    r = analyse(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32), SR)
    assert r.responses == [] and len(r.warnings) == 2


# ------------------------------------------------------------------ dead air
def test_dead_air_needs_3_seconds_of_silence_on_both_channels_between_speech():
    r = run([("customer", 1.0, 3.0), ("agent", 5.5, 7.0), ("customer", 11.0, 12.0)])
    assert [(round(a), round(b)) for a, b in r.dead_air] == [(7, 11)]                # 2.5 s (3 to 5.5) is not enough
    assert r.pct(50) == pytest.approx(2.5, abs=0.1)


def test_silence_before_the_first_and_after_the_last_speech_is_not_dead_air():
    assert run([("customer", 8.0, 9.0), ("agent", 9.5, 10.5)], seconds=30.0).dead_air == []


# ------------------------------------------------------------------ files and command line
def test_agent_channel_choice_swaps_the_roles(tmp_path):
    p = write(tmp_path / "c.wav", make_call([("customer", 1.0, 3.0), ("agent", 3.8, 6.0)]))
    assert analyse_file(p, agent_channel=0).responses[0] == pytest.approx(0.8, abs=0.06)
    r = analyse_file(p, agent_channel=1)                                             # channels swapped: the "agent" now speaks first
    assert r.responses == [] and r.events[0].kind == "customer_pause"


def test_two_mono_files_work_like_one_stereo_file(tmp_path):
    x = make_call([("customer", 1.0, 3.0), ("agent", 3.8, 6.0)])
    a = write(tmp_path / "a.wav", x[:, 0])
    c = write(tmp_path / "c.wav", x[:, 1])
    assert analyse_file(a, customer=c).responses == analyse_file(write(tmp_path / "s.wav", x)).responses


def test_wrong_inputs_are_clear_errors(tmp_path):
    mono = write(tmp_path / "m.wav", make_call([("customer", 1.0, 3.0)])[:, 0])
    with pytest.raises(CallError, match="two-channel"):
        analyse_file(mono)
    with pytest.raises(CallError, match="cannot read"):
        analyse_file(tmp_path / "nope.wav")
    (tmp_path / "bad.wav").write_bytes(b"junk")
    with pytest.raises(CallError):
        analyse_file(tmp_path / "bad.wav")
    other = write(tmp_path / "o.wav", make_call([("customer", 1.0, 3.0)])[:, 0], 8000)
    with pytest.raises(CallError, match="sample rates"):
        analyse_file(mono, customer=other)


def test_cli_prints_a_report_and_writes_json_and_csv(tmp_path, capsys):
    write(tmp_path / "call1.wav", make_call([("customer", 1.0, 3.0), ("agent", 3.8, 6.0), ("customer", 7.0, 8.0), ("agent", 8.5, 10.0)]))
    write(tmp_path / "call2.wav", make_call([("customer", 1.0, 3.0), ("agent", 5.0, 6.0)]))
    assert main(["analyze", str(tmp_path), "--json", str(tmp_path / "o.json"), "--csv", str(tmp_path / "o.csv")]) == 0
    out = capsys.readouterr().out
    assert "call1.wav" in out and "call2.wav" in out and "agent response delay (2 turns)" in out
    j = json.loads((tmp_path / "o.json").read_text())
    assert [c["file"] for c in j] == ["call1.wav", "call2.wav"] and len(j[0]["response_delays_s"]) == 2
    assert j[0]["p90_response_s"] > j[0]["median_response_s"] + 0.05
    rows = list(csv.DictReader(open(tmp_path / "o.csv")))
    assert [r["file"] for r in rows] == ["call1.wav", "call2.wav"] and float(rows[1]["median_response_s"]) == pytest.approx(2.0, abs=0.1)


def test_cli_response_limits_set_the_exit_code(tmp_path, capsys):
    p = write(tmp_path / "slow.wav", make_call([("customer", 1.0, 3.0), ("agent", 5.0, 6.0), ("customer", 8.0, 9.0), ("agent", 11.0, 12.0)]))
    assert main(["analyze", str(p)]) == 0
    assert main(["analyze", str(p), "--max-median-response", "3.0"]) == 0
    assert main(["analyze", str(p), "--max-median-response", "1.0"]) == 1
    assert main(["analyze", str(p), "--max-p90-response", "1.0"]) == 1
    assert "above the limit" in capsys.readouterr().out


def test_cli_errors(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "nope")]) == 2
    assert main(["analyze", str(tmp_path)]) == 2                                      # empty folder
    write(tmp_path / "mono.wav", make_call([("customer", 1.0, 3.0)])[:, 0])
    assert main(["analyze", str(tmp_path / "mono.wav")]) == 1                          # nothing could be analysed
    assert "ERROR" in capsys.readouterr().err


def test_a_barge_in_through_unrelated_speech_does_not_fake_a_leak():
    """The agent talking all the way through the customer's few words is overlap, not leakage."""
    r = run([("agent", 1.0, 8.0), ("customer", 4.0, 5.5)])
    assert r.leak_customer_to_agent_db is None and r.warnings == []


@pytest.mark.parametrize("bleed", [-25.0, -18.0])
def test_leakage_through_a_room_is_not_mistaken_for_speech(bleed):
    r = run([("customer", 1.0, 3.0), ("agent", 3.8, 6.0), ("customer", 7.0, 8.0), ("agent", 8.6, 10.0)], bleed_db=bleed, smear=True)
    assert r.customer_speech_s == pytest.approx(3.0, abs=0.5) and r.agent_speech_s == pytest.approx(3.6, abs=0.5)
    assert r.interrupts == 0 and r.barge_ins == 0 and len(r.responses) == 2


# ------------------------------------------------------------------ detector edges (found by planting bugs)
def test_a_quiet_steady_hum_far_below_the_speech_is_not_speech():
    hum = (0.0018 * np.sin(2 * np.pi * 120 * np.arange(SR * 12) / SR)).astype(np.float32)          # about -55 dBFS, always on
    x = make_call([("agent", 3.0, 4.5), ("customer", 6.0, 9.0)], seconds=12.0, noise_db=-80.0)
    r = analyse(x[:, 0] + hum, x[:, 1], SR)
    assert r.agent_speech_s == pytest.approx(1.5, abs=0.3)


@pytest.mark.parametrize("level", [-75.0, -62.0, -46.0])
def test_a_channel_with_only_extremely_quiet_sound_has_no_speech(level):
    quiet = make_call([("agent", 1.0, 3.0), ("agent", 4.0, 6.0)], noise_db=-100.0, level_db=level)[:, 0]
    loud = make_call([("customer", 1.0, 3.0), ("customer", 4.0, 6.0)])[:, 1]
    assert analyse(quiet, loud, SR).agent_speech_s < 0.2


def test_short_gaps_are_bridged_and_short_bursts_are_dropped():
    def m(*runs, n=400):
        a = np.zeros(n, dtype=bool)
        for s, e in runs:
            a[s:e] = True
        return a
    assert len(vad.bridge_and_drop(m((10, 100), (110, 200)))) == 1               # a 0.10 s gap is bridged (between words)
    assert len(vad.bridge_and_drop(m((10, 100), (130, 200)))) == 2               # a 0.30 s gap is not
    assert vad.bridge_and_drop(m((10, 15))) == []                                # a 50 ms burst is a click
    assert len(vad.bridge_and_drop(m((10, 20)))) == 1                            # 100 ms is a word


def test_unrelated_sound_on_the_other_channel_is_not_reported_as_a_leak():
    """A leak follows the loud voice; separate quiet talking on the other side does not."""
    base = make_call([("customer", 1.0, 9.0)], seconds=10.0)
    from synth import utterance
    other = np.zeros(len(base))
    for k, (s, e) in enumerate([(1.5, 3.0), (4.0, 5.2), (6.5, 8.5)]):
        u = utterance(s, e, 50 + k)
        other[int(s * SR):int(s * SR) + len(u)] += 0.05 * u
    r = analyse((base[:, 0] + other).astype(np.float32), base[:, 1], SR)
    assert r.leak_customer_to_agent_db is None


def test_a_leak_through_a_room_is_still_recognised_as_a_leak():
    r = run([("customer", 1.0, 4.0), ("agent", 5.0, 8.0), ("customer", 9.0, 11.0)], bleed_db=-9.0, smear=True)
    assert any("strong leakage" in w for w in r.warnings)


def test_too_little_loud_speech_is_not_enough_to_measure_a_leak():
    r = run([("agent", 1.0, 1.5), ("customer", 3.0, 7.0)], bleed_db=-9.0)
    assert r.leak_agent_to_customer_db is None


def test_quiet_sporadic_sounds_far_below_the_speech_are_not_speech():
    """Quiet chatter or clicks in the background (about -50 dBFS) while real speech is near -12 dBFS."""
    from synth import utterance
    x = make_call([("agent", 3.0, 4.5), ("customer", 8.0, 11.0)], seconds=14.0, noise_db=-80.0)
    bg = np.zeros(len(x))
    for k, (s, e) in enumerate([(0.5, 2.0), (5.0, 7.0), (11.5, 13.5)]):
        v = utterance(s, e, 70 + k)
        bg[int(s * SR):int(s * SR) + len(v)] += 0.004 * v
    r = analyse((x[:, 0] + bg).astype(np.float32), x[:, 1], SR)
    assert r.agent_speech_s == pytest.approx(1.5, abs=0.3)
