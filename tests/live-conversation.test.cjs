const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

function fixture(extra = {}) {
  const context = vm.createContext({
    console,
    Float32Array,
    DOMException,
    document: { hidden: false },
    ...extra,
  });
  vm.runInContext(fs.readFileSync("web/live-conversation.js", "utf8"), context);
  const Type = vm.runInContext("LiveConversation", context);
  const turns = [],
    states = [];
  let speech = 0,
    talking = false;
  const live = new Type({
    onState: (s) => states.push(s),
    onLevel: () => {},
    onSpeechStart: () => speech++,
    onTurn: (parts, rate) => turns.push({ parts, rate }),
    isAssistantTalking: () => talking,
  });
  live.active = true;
  live.noise = 0.001;
  live.context = { sampleRate: 16000, close: async () => {} };
  const feed = (rms, blocks = 1) => {
    for (let i = 0; i < blocks; i++)
      live.accept(
        Float32Array.from(
          { length: 1600 },
          (_, i) =>
            Math.sqrt(2) * rms * Math.sin((2 * Math.PI * 220 * i) / 16000),
        ),
      );
  };
  return {
    live,
    feed,
    turns,
    states,
    context,
    get speech() {
      return speech;
    },
    setTalking: (value) => (talking = value),
  };
}

test("continuous silence never submits; two spoken turns need no restart", () => {
  const f = fixture();
  f.feed(0.001, 1000);
  for (let i = 0; i < 50; i++) f.live.accept(new Float32Array(1600).fill(0.02));
  assert.equal(f.turns.length, 0);
  for (let i = 0; i < 2; i++) {
    f.feed(0.1, 5);
    f.feed(0.001, 8);
  }
  assert.equal(f.speech, 2);
  assert.equal(f.turns.length, 2);
  assert.equal(f.live.active, true);
  assert.ok(f.turns[0].parts.reduce((n, c) => n + c.length, 0) >= 16000 * 0.5);
});

test("brief impulse and quiet playback leakage do not cause barge-in", () => {
  const f = fixture();
  f.setTalking(true);
  f.feed(0.1);
  f.feed(0.001, 10);
  f.feed(0.015, 20);
  assert.equal(f.speech, 0);
  f.feed(0.1, 5);
  f.feed(0.001, 8);
  assert.equal(f.speech, 1);
  assert.equal(f.turns.length, 1);
});

test("End during AudioContext startup immediately releases acquired tracks", async () => {
  let release,
    stopped = 0;
  const track = { stop: () => stopped++ };
  const media = { getTracks: () => [track] };
  class AudioContext {
    constructor() {
      this.audioWorklet = { addModule: async () => {} };
    }
    resume() {
      return new Promise((resolve) => (release = resolve));
    }
    async close() {}
  }
  const f = fixture({
    navigator: { mediaDevices: { getUserMedia: async () => media } },
    AudioContext,
  });
  f.live.active = false;
  const starting = f.live.start();
  const rejected = assert.rejects(starting, { name: "AbortError" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(typeof release, "function");
  await f.live.stop();
  assert.ok(stopped >= 1);
  release();
  await rejected;
  assert.equal(f.live.active, false);
});

test("leaving and returning during permission does not activate microphone later", async () => {
  let allow,
    stopped = 0;
  const f = fixture({
    navigator: {
      mediaDevices: {
        getUserMedia: () => new Promise((resolve) => (allow = resolve)),
      },
    },
  });
  f.live.active = false;
  const starting = f.live.start();
  f.context.document.hidden = true;
  await f.live.stop();
  f.context.document.hidden = false;
  allow({ getTracks: () => [{ stop: () => stopped++ }] });
  await assert.rejects(starting, { name: "AbortError" });
  assert.equal(stopped, 1);
  assert.equal(f.live.active, false);
});

test("queued audio cannot cross End then Start session boundaries", async () => {
  const source = fs.readFileSync("web/app.js", "utf8");
  const body = source.slice(
    source.indexOf("async function sendLiveTurn("),
    source.indexOf("async function startLiveConversation("),
  );
  let sent = 0;
  const context = vm.createContext({
    state: { busy: true, liveUtterance: 1 },
    live: { active: true, generation: 1 },
    performance,
    setTimeout,
    notice: () => {},
    wavBlob: () => null,
    submitTurn: async () => sent++,
  });
  vm.runInContext(body, context);
  const pending = vm.runInContext("sendLiveTurn([],16000)", context);
  context.live.active = false;
  context.live.generation++;
  context.live.active = true;
  context.state.busy = false;
  await pending;
  assert.equal(sent, 0);
});
