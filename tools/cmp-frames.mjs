// 開発用: 2 つの変換結果 (frames.json) を時刻でそろえて、メロディ・ベース・ドラムの一致率を出す
//   node tools/cmp-frames.mjs <目標の frames.json> <比べる frames.json> [開始秒] [終了秒]
import fs from 'node:fs';
const [aFile, bFile, t0 = '2', t1 = '9999'] = process.argv.slice(2);
const A = JSON.parse(fs.readFileSync(aFile)), B = JSON.parse(fs.readFileSync(bFile));
const bm = new Map(B.map((f) => [Math.round(f.t * 100), f]));
const S = { mel: [0, 0, 0], melOn: [0, 0, 0], bass: [0, 0, 0], drum: [0, 0] };
for (const a of A) {
  if (a.t < +t0 || a.t > +t1) continue;
  const k = Math.round(a.t * 100);
  const b = bm.get(k) || bm.get(k + 1) || bm.get(k - 1);
  if (!b) continue;
  const x = a.p1.note, y = b.p1.note;
  // 発声の一致 (鳴っている / 鳴っていない)
  S.melOn[0]++; if ((x >= 0) === (y >= 0)) S.melOn[1]++;
  if (x >= 0 && y >= 0) { S.mel[0]++; if (x === y) S.mel[1]++; if (x % 12 === y % 12) S.mel[2]++; }
  if (a.tri >= 0 && b.tri >= 0) { S.bass[0]++; if (a.tri === b.tri) S.bass[1]++; if (a.tri % 12 === b.tri % 12) S.bass[2]++; }
  if (a.drum) { S.drum[0]++; if (a.drum === b.drum) S.drum[1]++; }
}
const p = (x, n) => (n ? ((100 * x) / n).toFixed(1) + '%' : '-');
console.log(`メロディ 音程 ${p(S.mel[1], S.mel[0])} 音名 ${p(S.mel[2], S.mel[0])} | 発声の一致 ${p(S.melOn[1], S.melOn[0])} | ベース 音程 ${p(S.bass[1], S.bass[0])} 音名 ${p(S.bass[2], S.bass[0])} | ドラム ${p(S.drum[1], S.drum[0])}`);
