// ============================================================
//  Standard MIDI File (SMF Format 1) 書き出し (依存なし・純粋 JS)
//  ---------------------------------------------------------
//  フル曲高精度 MIDI アレンジの結果を、DAW / FamiTracker / Domino 等で
//  そのまま開ける .mid ファイルとして出力する。
//    Track 0: テンポ / キー / 拍子マップ
//    Track 1: Pulse 1 (メロディ, Ch 0, Square Lead #80)
//    Track 2: Pulse 2 (伴奏 / アルペジオ, Ch 1, Square Lead #80)
//    Track 3: Triangle (ベース, Ch 2, Synth Bass #38)
//    Track 4: Drums (DPCM / Noise, Ch 9 GM Drums)
// ============================================================

const TPQ = 480; // 1 拍 = 480 ticks (16 分音符 = 120 ticks)

function vlq(n) {
  n = Math.max(0, Math.round(n));
  const out = [n & 0x7f];
  n >>= 7;
  while (n > 0) {
    out.push((n & 0x7f) | 0x80);
    n >>= 7;
  }
  return out.reverse();
}

function u16(v) { return [(v >> 8) & 0xff, v & 0xff]; }
function u32(v) { return [(v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff]; }
function ascii(s) { return Array.from(s, (c) => c.charCodeAt(0) & 0x7f); }

function makeTrack(events) {
  // 同時刻では NoteOff (0x80) を NoteOn (0x90) より先に並べる
  events.sort((a, b) => a.tick - b.tick || (a.ord || 0) - (b.ord || 0));
  const body = [];
  let prev = 0;
  for (const ev of events) {
    const dt = Math.max(0, Math.round(ev.tick) - prev);
    prev = Math.max(prev, Math.round(ev.tick));
    body.push(...vlq(dt), ...ev.bytes);
  }
  body.push(0x00, 0xff, 0x2f, 0x00); // End of Track
  return [...ascii('MTrk'), ...u32(body.length), ...body];
}

// 調号のシャープ/フラット数 (-7..+7)
const MAJ_SF = [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5];
const MIN_SF = [-3, 4, -1, 6, 1, -4, 3, -2, 5, 0, -5, 2];

// arrangement: { bpm, beats, keyRoot, keyMinor, tracks: { p1, p2, tri, drums } }
//   各ノート: { startBeat, endBeat, note, vel } (vel: 1..127)
//   ドラム:   { beat, kind: 'kick'|'snare'|'hat'|'openhat'|'crash', vel }
export function exportMidi({ bpm = 120, keyRoot = 0, keyMinor = false, tracks }) {
  const tempoEvents = [
    { tick: 0, ord: 0, bytes: [0xff, 0x58, 0x04, 4, 2, 24, 8] }, // 4/4
    { tick: 0, ord: 1, bytes: [0xff, 0x51, 0x03, ...u32(Math.round(60000000 / Math.max(40, bpm))).slice(1)] },
  ];
  if (keyRoot >= 0) {
    const sf = (keyMinor ? MIN_SF : MAJ_SF)[keyRoot % 12] & 0xff;
    tempoEvents.push({ tick: 0, ord: 2, bytes: [0xff, 0x59, 0x02, sf, keyMinor ? 1 : 0] });
  }

  const buildNoteTrack = (name, ch, program, notes) => {
    const evs = [
      { tick: 0, ord: 0, bytes: [0xff, 0x03, name.length, ...ascii(name)] },
      { tick: 0, ord: 1, bytes: [0xc0 | ch, program & 0x7f] },
    ];
    for (const n of notes || []) {
      const pitch = Math.max(0, Math.min(127, Math.round(n.note)));
      const vel = Math.max(1, Math.min(127, Math.round(n.vel || 100)));
      const t0 = Math.max(0, Math.round(n.startBeat * TPQ));
      const t1 = Math.max(t0 + 15, Math.round(n.endBeat * TPQ));
      evs.push({ tick: t0, ord: 10, bytes: [0x90 | ch, pitch, vel] });
      evs.push({ tick: t1, ord: 5, bytes: [0x80 | ch, pitch, 0] });
    }
    return makeTrack(evs);
  };

  const DRUM_NOTE = { kick: 36, snare: 38, hat: 42, openhat: 46, crash: 49 };
  const drumEvs = [{ tick: 0, ord: 0, bytes: [0xff, 0x03, 5, ...ascii('Drums')] }];
  for (const d of tracks.drums || []) {
    const pitch = DRUM_NOTE[d.kind] ?? 42;
    const vel = Math.max(1, Math.min(127, Math.round(d.vel || 96)));
    const t0 = Math.max(0, Math.round(d.beat * TPQ));
    drumEvs.push({ tick: t0, ord: 10, bytes: [0x99, pitch, vel] });
    drumEvs.push({ tick: t0 + 60, ord: 5, bytes: [0x89, pitch, 0] });
  }

  const allTracks = [
    makeTrack(tempoEvents),
    buildNoteTrack('Pulse 1 (Melody)', 0, 80, tracks.p1),
    buildNoteTrack('Pulse 2 (Accomp)', 1, 80, tracks.p2),
    buildNoteTrack('Triangle (Bass)', 2, 38, tracks.tri),
    makeTrack(drumEvs),
  ];

  const header = [...ascii('MThd'), ...u32(6), ...u16(1), ...u16(allTracks.length), ...u16(TPQ)];
  const total = header.length + allTracks.reduce((s, t) => s + t.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  out.set(header, off); off += header.length;
  for (const t of allTracks) { out.set(t, off); off += t.length; }
  return out;
}
