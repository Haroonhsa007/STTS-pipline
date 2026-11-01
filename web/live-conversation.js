"use strict";

// One microphone connection per session. Only detected utterances leave the
// browser; silence never reaches ASR. Browser echo cancellation remains enabled
// while replies play, so the caller can interrupt by speaking.
class LiveConversation {
  constructor(callbacks) {
    this.callbacks = callbacks;
    this.active = false;
    this.generation = 0;
    this.resetUtterance();
  }
  resetUtterance() {
    this.preRoll = [];
    this.parts = [];
    this.capturing = false;
    this.voiced = 0;
    this.quiet = 0;
    this.duration = 0;
    this.onset = 0;
  }
  async start(deviceId) {
    const generation = ++this.generation;
    const check = () => {
      if (generation !== this.generation || document.hidden)
        throw new DOMException("Conversation start cancelled.", "AbortError");
    };
    let media, context;
    try {
      media = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      check();
      this.media = media;
      context = new AudioContext();
      this.context = context;
      await context.resume();
      check();
      await context.audioWorklet.addModule("/static/recorder-worklet.js");
      check();
      if (generation !== this.generation || document.hidden) {
        media.getTracks().forEach((track) => track.stop());
        await context.close();
        return;
      }
      this.media = media;
      this.context = context;
      this.source = context.createMediaStreamSource(media);
      this.worklet = new AudioWorkletNode(context, "pcm-recorder");
      this.silent = context.createGain();
      this.silent.gain.value = 0;
      this.source.connect(this.worklet);
      this.worklet.connect(this.silent);
      this.silent.connect(context.destination);
      this.worklet.port.postMessage({ continuous: true });
      this.worklet.port.onmessage = ({ data }) => {
        if (data instanceof Float32Array) this.accept(data);
      };
      this.noise = 0.001;
      this.resetUtterance();
      this.active = true;
      this.callbacks.onState("Listening");
    } catch (error) {
      media?.getTracks().forEach((track) => track.stop());
      await context?.close().catch(() => {});
      throw error;
    }
  }
  accept(samples) {
    if (!this.active) return;
    const seconds = samples.length / this.context.sampleRate;
    const mean =
      samples.reduce((sum, value) => sum + value, 0) / samples.length;
    const rms = Math.sqrt(
      samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
        samples.length,
    );
    const assistantTalking = this.callbacks.isAssistantTalking();
    const threshold = Math.max(
      assistantTalking ? 0.024 : 0.012,
      Math.min(0.04, this.noise * 4),
    );
    this.callbacks.onLevel(samples, rms, this.duration);
    if (!this.capturing) {
      this.preRoll.push(samples);
      while (
        this.preRoll.reduce((n, chunk) => n + chunk.length, 0) >
        this.context.sampleRate * 0.45
      )
        this.preRoll.shift();
      if (rms >= threshold) this.onset += seconds;
      else {
        this.onset = 0;
        if (!assistantTalking) this.noise = this.noise * 0.98 + rms * 0.02;
      }
      if (this.onset < (assistantTalking ? 0.3 : 0.12)) return;
      this.parts = [...this.preRoll];
      this.duration =
        this.parts.reduce((n, chunk) => n + chunk.length, 0) /
        this.context.sampleRate;
      this.voiced = this.onset;
      this.capturing = true;
      this.quiet = 0;
      this.callbacks.onState("You are speaking");
      this.callbacks.onSpeechStart();
      return;
    }
    this.parts.push(samples);
    this.duration += seconds;
    if (rms >= threshold) {
      this.voiced += seconds;
      this.quiet = 0;
    } else this.quiet += seconds;
    if (this.quiet < 0.75 && this.duration < 29) return;
    const parts = this.parts;
    const voiced = this.voiced;
    this.resetUtterance();
    this.callbacks.onState("Listening");
    if (voiced >= 0.14) this.callbacks.onTurn(parts, this.context.sampleRate);
  }
  async stop() {
    this.generation++;
    this.active = false;
    this.worklet?.disconnect();
    this.source?.disconnect();
    this.silent?.disconnect();
    this.media?.getTracks().forEach((track) => track.stop());
    this.resetUtterance();
    this.callbacks.onState("Off");
    await this.context?.close().catch(() => {});
  }
}
