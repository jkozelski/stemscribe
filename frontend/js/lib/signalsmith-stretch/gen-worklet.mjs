// Generate a STATIC, same-origin worklet .js file from the vendored Signalsmith lib.
// The lib normally builds the worklet module as a Blob URL (blocked by our CSP
// script-src which has no blob:). We capture the EXACT module code the lib would
// have put in the blob and write it to a real .js file served from 'self'.
//
// Strategy: import the lib in Node, stub just enough Web Audio so createNode()
// runs to the point where it constructs `moduleCode` and calls
// URL.createObjectURL(new Blob([moduleCode])). We intercept that call.
import { writeFileSync } from 'fs';

const LIB = process.argv[2] ||
  '/Users/jeffkozelski/stemscribe/frontend/js/lib/signalsmith-stretch/SignalsmithStretch.mjs';
const OUT = process.argv[3] ||
  '/Users/jeffkozelski/stemscribe/frontend/js/lib/signalsmith-stretch/SignalsmithStretch.worklet.js';

let captured = null;

// Minimal global stubs so the browser-side createNode path runs in Node.
globalThis.window = globalThis;
globalThis.document = { currentScript: { src: '' } };
globalThis.URL = globalThis.URL || {};
globalThis.URL.createObjectURL = (blob) => {
  // blob is our fake Blob; pull the text out.
  captured = blob.__text;
  return 'about:blank#captured';
};
class FakeBlob {
  constructor(parts) { this.__text = parts.join(''); }
}
globalThis.Blob = FakeBlob;

// AudioWorkletNode: throw on first construction so the lib falls into the
// "build module + addModule" branch (that's where moduleCode is created).
let threwOnce = false;
globalThis.AudioWorkletNode = class {
  constructor() {
    if (!threwOnce) { threwOnce = true; throw new Error('force-module-build'); }
    // second construction (after addModule) — return a dummy; we never reach audio.
    this.port = { postMessage(){}, onmessage:null, close(){} };
  }
};

const mod = await import(LIB);
const SignalsmithStretch = mod.default;

const fakeCtx = {
  audioWorklet: {
    addModule: async (url) => { /* no-op: we already captured moduleCode */ }
  },
};

try {
  // This drives createNode -> AudioWorkletNode throws -> moduleCode built ->
  // URL.createObjectURL intercepted -> captured set -> addModule no-op ->
  // AudioWorkletNode constructed again (dummy) -> then it awaits a 'ready'
  // port message that never comes. We don't care; we already captured.
  const p = SignalsmithStretch(fakeCtx);
  // Give the microtask queue a tick, then bail.
  await Promise.race([p, new Promise(r => setTimeout(r, 200))]);
} catch (e) {
  // expected — we only needed the moduleCode capture
}

if (!captured) {
  console.error('FAILED: did not capture worklet module code');
  process.exit(1);
}

const header =
  '// AUTO-GENERATED from SignalsmithStretch.mjs by gen-worklet.mjs.\n' +
  '// Static same-origin AudioWorklet module (CSP script-src \'self\' safe;\n' +
  '// replaces the lib\'s default blob: URL which our CSP blocks).\n' +
  '// Do not hand-edit — regenerate if the vendored lib changes.\n';

writeFileSync(OUT, header + captured + '\n');
console.log('WROTE ' + OUT + ' (' + (header.length + captured.length) + ' bytes)');
console.log('head: ' + captured.slice(0, 90));
