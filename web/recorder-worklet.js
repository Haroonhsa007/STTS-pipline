// Batch PCM frames off the main thread. Audio output stays silent to avoid feedback.
class PCMRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.used = 0;
    this.total = 0;
    this.limit = Math.floor(sampleRate * 29.5);
    this.done = false;
    this.port.onmessage = ({ data }) => {
      if (data?.continuous) this.limit = Infinity;
      if (data === "flush") {
        if (this.used) this.port.postMessage(this.buffer.slice(0, this.used));
        this.used = 0;
        this.port.postMessage("flushed");
      }
    };
  }
  process(inputs) {
    if (this.done) return true;
    const channels = inputs[0];
    if (channels?.length) {
      for (let i = 0; i < channels[0].length; i++) {
        let sample = 0;
        for (const channel of channels) sample += channel[i] / channels.length;
        this.buffer[this.used++] = sample;
        this.total++;
        if (this.used === this.buffer.length) {
          this.port.postMessage(this.buffer, [this.buffer.buffer]);
          this.buffer = new Float32Array(2048);
          this.used = 0;
        }
        if (this.total >= this.limit) {
          this.done = true;
          this.port.postMessage("limit");
          break;
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm-recorder", PCMRecorder);
