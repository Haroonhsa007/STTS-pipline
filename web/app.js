"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  session: crypto.randomUUID(),
  busy: false,
  starting: false,
  startGeneration: 0,
  recording: null,
  results: [],
  stream: null,
  activeMessage: null,
  assistantMessage: null,
  lastDelta: 0,
  currentJob: null,
  streamedAudio: false,
  interrupted: false,
  activeConversation: false,
  bargeIn: false,
  liveUtterance: 0,
  preparing: false,
  prepared: false,
};
const secs = (value) =>
  Number.isFinite(value) ? `${value.toFixed(2)} s` : "—";
const number = (value, suffix = "", decimals = 1) =>
  Number.isFinite(value) ? `${value.toFixed(decimals)}${suffix}` : "—";
const displayNames = {
  load_bilingual: "Prepare conversation",
  translate_input: "Understand Urdu",
  translate_reply: "Render Urdu reply",
  tts_chunk: "Speech chunk",
  load_native: "Load native model",
  speech_to_speech: "Text & speech · shared model",
  input_prepare: "Prepare recording",
  asr: "Speech recognition",
  llm: "Language model",
  tts: "Speech synthesis",
  save_audio: "Save waveform",
  tts_verification: "Verify speech",
  voice_conditioning: "Prepare reference voice",
  verify_tts: "Verify speech",
};
function notice(text, error = false) {
  $("notice").textContent = text;
  $("notice").classList.toggle("error", error);
}
class StreamingPlayback {
  constructor() {
    this.context = null;
    this.sources = new Set();
    this.next = 0;
    this.generation = 0;
  }
  prepare() {
    try {
      this.context ||= new AudioContext();
      this.context
        .resume()
        .catch(() => notice("Press a playback control to enable audio."));
    } catch (error) {
      notice(`Audio playback unavailable: ${error.message}`, true);
    }
  }
  enqueue(encoded, rate) {
    this.prepare();
    if (!this.context) return;
    const bytes = Uint8Array.from(atob(encoded), (character) =>
      character.charCodeAt(0),
    );
    const view = new DataView(bytes.buffer),
      samples = new Float32Array(bytes.length / 2);
    for (let i = 0; i < samples.length; i++)
      samples[i] = view.getInt16(i * 2, true) / 32768;
    const buffer = this.context.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(samples, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    const start = Math.max(this.next, this.context.currentTime + 0.08),
      generation = this.generation;
    this.next = start + buffer.duration;
    this.sources.add(source);
    if (
      this.context.state === "running" &&
      state.browserMetrics?.scheduled_first_audio_s == null
    )
      state.browserMetrics.scheduled_first_audio_s =
        (performance.now() - state.submittedAt) / 1000 +
        (start - this.context.currentTime);
    source.onended = () => {
      this.sources.delete(source);
      source.disconnect();
      controls();
      if (generation === this.generation) maybeContinue();
    };
    source.start(start);
    controls();
  }
  stop() {
    this.generation++;
    for (const source of this.sources) {
      try {
        source.stop();
        source.disconnect();
      } catch {}
    }
    this.sources.clear();
    this.next = 0;
    $("player").pause();
  }
}
const playback = new StreamingPlayback();
const live = new LiveConversation({
  isAssistantTalking: () => playback.sources.size > 0 || !$("player").paused,
  onState: (label) => {
    $("mic-state").textContent = label;
  },
  onLevel: (samples, rms, duration) => {
    paintWave(samples);
    $("rms").textContent = rms.toFixed(5);
    $("record-time").textContent = duration
      ? `${duration.toFixed(1)} s`
      : "Listening continuously";
  },
  onSpeechStart: () => {
    state.liveUtterance++;
    if (
      !state.preparing &&
      (state.busy || playback.sources.size || !$("player").paused)
    )
      cancelReply(true);
  },
  onTurn: (parts, rate) => sendLiveTurn(parts, rate),
});

async function sendLiveTurn(parts, rate) {
  const generation = state.liveUtterance;
  const microphoneGeneration = live.generation;
  const deadline = performance.now() + 70000;
  while (state.busy && live.active && performance.now() < deadline)
    await new Promise((resolve) => setTimeout(resolve, 50));
  if (
    !live.active ||
    generation !== state.liveUtterance ||
    microphoneGeneration !== live.generation
  )
    return;
  if (state.busy) {
    notice(
      "The previous turn has not stopped. End the session and try again.",
      true,
    );
    return;
  }
  await submitTurn({ audio: wavBlob(parts, rate) });
}

async function startLiveConversation() {
  state.starting = true;
  state.activeConversation = true;
  playback.prepare();
  controls();
  try {
    await live.start($("mic-device").value);
    if (!live.active) return;
    notice(
      "Listening. Speak naturally; you can interrupt the reply by speaking.",
    );
    if (!state.prepared && $("engine").value !== "python") {
      state.starting = false;
      await submitTurn({ prepare: true });
    }
  } catch (error) {
    state.activeConversation = false;
    notice(error.message, error.name !== "AbortError");
  } finally {
    state.starting = false;
    controls();
  }
}

async function cancelReply(keepListening = false) {
  state.interrupted = true;
  state.bargeIn = keepListening;
  if (!keepListening) {
    state.activeConversation = false;
    state.startGeneration++;
    if (state.recording) stopRecording(true);
    await live.stop();
    paintWave();
  }
  playback.stop();
  controls();
  if (state.busy && state.currentJob) {
    try {
      await responseJSON(
        await fetch(
          `/api/jobs/${state.currentJob}/cancel?session=${state.session}`,
          { method: "POST" },
        ),
      );
    } catch (error) {
      notice(error.message, true);
    }
  }
}

function maybeContinue() {
  if (live.active) return;
  if (
    state.activeConversation &&
    $("continuous").checked &&
    !document.hidden &&
    !state.busy &&
    !state.recording &&
    !state.starting &&
    !playback.sources.size &&
    state.results.at(-1)?.status === "ok"
  )
    startRecording(false);
}
function controls() {
  const locked =
    state.busy || state.starting || Boolean(state.recording) || live.active;
  document
    .querySelectorAll(
      ".settings input,.settings textarea,.settings select,#mode,#reset,#audio-file,#mic-device",
    )
    .forEach((el) => (el.disabled = locked));
  $("send").disabled = locked;
  $("warmup").disabled = locked || $("engine").value === "python";
  $("message").disabled = locked;
  $("interrupt").disabled = !(
    state.busy ||
    playback.sources.size ||
    state.recording ||
    state.activeConversation
  );
  $("interrupt").textContent = state.recording
    ? "End conversation"
    : "Interrupt";
  $("interrupt").hidden = live.active;
  $("record").disabled =
    !live.active &&
    (state.busy || state.starting || state.recording?.test === true);
  $("check-mic").disabled =
    state.busy ||
    state.starting ||
    state.recording?.test === false ||
    live.active;
  $("record").textContent = live.active
    ? "End conversation"
    : state.recording?.test === false
      ? "Stop & send"
      : "Start conversation";
  $("record").classList.toggle(
    "recording",
    live.active || state.recording?.test === false,
  );
  $("continuous").disabled = locked;
  $("auto-stop").closest("label").hidden = $("continuous").checked;
  $("check-mic").textContent = state.recording?.test
    ? "Stop check"
    : "Check microphone";
}
function configureVoice() {
  const bilingual = $("engine").value === "bilingual";
  if ($("language").value === "auto" && !bilingual)
    $("engine").value = "bilingual";
  const urdu = $("language").value === "ur";
  const liquidOption = $("tts").querySelector('[value="liquid"]');
  liquidOption.disabled = urdu;
  if (urdu && $("tts").value === "liquid") $("tts").value = "omnivoice";
  if ($("engine").value === "bilingual") $("tts").value = "omnivoice";
  else if (urdu || $("tts").value !== "liquid") $("engine").value = "python";
  const native = $("engine").value === "native";
  const tts = $("tts").value;
  $("liquid-controls").hidden = native || tts !== "liquid";
  $("omni-controls").hidden = tts !== "omnivoice";
  $("mio-controls").hidden = tts !== "indic-mio";
  $("warmup").hidden = $("engine").value === "python";
  $("native-ready").hidden = $("engine").value === "python";
  $("reference-controls").hidden = tts === "liquid";
  $("reference-text-field").hidden = tts !== "omnivoice";
  $("steps-field").hidden = tts !== "omnivoice";
  $("expression").closest(".field").hidden = $("engine").value === "bilingual";
  if ($("engine").value === "bilingual") $("expression").value = "neutral";
  $("tokens").closest(".field").hidden = native && $("mode").value === "run";
  $("voice-note").textContent = {
    liquid: "Built-in voices. Gender and English locale controls.",
    omnivoice:
      "Describe a voice or provide a reference clip. Vocal events are not emotion scores.",
    "indic-mio":
      "Reference clip required. Named emotion tags vary by language.",
  }[tts];
  if (tts === "indic-mio") $("reference-controls").open = true;
  const emotions = urdu
    ? ["neutral", "happy", "sad", "angry", "disgust", "fear", "surprise"]
    : ["neutral", "happy", "sad", "enunciated", "confused", "angry", "whisper"];
  const previous = $("emotion").value;
  $("emotion").replaceChildren(
    ...emotions.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name[0].toUpperCase() + name.slice(1);
      return option;
    }),
  );
  if (emotions.includes(previous)) $("emotion").value = previous;
  if (native)
    $("voice-note").textContent =
      "One quantized 2B model streams English text and speech. Conversation uses its default voice.";
  $("asr-model").textContent = urdu
    ? "Indic-Transcribe Core · 1.2B"
    : "LFM3-Audio-2B";
  $("llm-model").textContent = native
    ? "Shared LFM3-Audio-2B · Q4"
    : urdu
      ? "Qwen3.5-2B"
      : "LFM3-1.7B";
  if ($("engine").value === "bilingual") {
    $("asr-model").textContent = "Indic Flex · 1.2B INT8";
    $("llm-model").textContent = "LFM3 text · Q4 + NLLB · 0.6B INT8";
  }
  $("access-note").textContent =
    $("engine").value === "bilingual"
      ? "English and Urdu are detected each turn. Conversation history is preserved."
      : urdu
        ? "Indic ASR needs approved Hugging Face access. Models load on the first turn."
        : "Models load when you send a turn.";
}
function settings() {
  const backend = $("tts").value;
  return {
    engine: $("engine").value,
    language: $("language").value,
    tts_backend: backend,
    device: $("device").value,
    gender: $("gender").value,
    locale: $("locale").value,
    emotion: backend === "indic-mio" ? $("emotion").value : "neutral",
    expression: backend === "omnivoice" ? $("expression").value : "neutral",
    speed: backend === "omnivoice" ? Number($("speed").value) : 1,
    voice_description:
      backend === "omnivoice" ? $("description").value.trim() : "",
    reference_text:
      backend === "omnivoice" && $("voice-file").files.length
        ? $("reference-text").value.trim()
        : "",
    max_new_tokens: Number($("tokens").value),
    omni_steps: Number($("steps").value),
    offline: $("offline").checked,
    verify_tts: $("verify").checked,
  };
}
function addMessage(role, text) {
  $("empty-chat")?.remove();
  const element = document.createElement("div");
  element.className = `message ${role}`;
  const name = document.createElement("span");
  name.className = "speaker";
  name.textContent = role === "assistant" ? "Assistant" : "You";
  const content = document.createElement("div");
  content.dir = "auto";
  content.textContent = text;
  element.append(name, content);
  $("messages").append(element);
  $("messages").scrollTop = $("messages").scrollHeight;
  return content;
}
function renderStages(stages) {
  const rows = Object.entries(stages);
  if (!rows.length) return;
  const max = Math.max(...rows.map(([, row]) => row.wall_s || 0), 0.001);
  const elements = rows.map(([name, row]) => {
    const element = document.createElement("div");
    element.className = `stage ${row.status}`;
    const label = document.createElement("div");
    label.className = "stage-label";
    const title = document.createElement("span");
    title.textContent =
      displayNames[row.stage_name] ||
      row.stage_name?.replaceAll("_", " ") ||
      name.replaceAll("_", " ");
    const value = document.createElement("span");
    value.textContent = `${secs(row.wall_s)}${row.status === "running" ? " · running" : row.status === "error" ? " · failed" : ""}`;
    const track = document.createElement("div");
    track.className = "stage-track";
    const fill = document.createElement("div");
    fill.className = "stage-fill";
    fill.style.width = `${(100 * (row.wall_s || 0)) / max}%`;
    track.append(fill);
    label.append(title, value);
    element.append(label, track);
    return element;
  });
  $("stages").replaceChildren(...elements);
}
function renderSystem(data) {
  $("native-memory").textContent = number(data.native_rss_mb / 1024, " GB", 2);
  $("free-disk").textContent = number(data.disk_free_mb / 1024, " GB", 2);
  $("cpu").textContent = number(data.system_cpu_percent, "%");
  $("memory").textContent = number(data.process_rss_mb / 1024, " GB", 2);
  $("available").textContent = number(
    data.system_available_mb / 1024,
    " GB",
    2,
  );
  $("swap").textContent = number(data.swap_used_mb / 1024, " GB", 2);
}
function renderMetrics(data) {
  $("phase").textContent =
    displayNames[data.phase] || data.phase || data.status;
  $("latency").textContent = secs(data.total_wall_s);
  for (const id of ["latency", "ttft", "first-audio"])
    $(id).classList.remove("missing");
  const stages = data.stages || {},
    llm = stages.llm || stages.speech_to_speech || {},
    asr = stages.asr || {},
    tts = stages.tts || stages.speech_to_speech || {};
  $("ttft").textContent = secs(llm.ttft_s);
  $("first-audio").textContent = secs(data.response_first_audio_s);
  $("token-speed").textContent = llm.token_scope?.startsWith("Native combined")
    ? "Shared text + audio"
    : number(llm.tokens_per_s, " tok/s");
  $("token-count").textContent = number(llm.output_tokens, "", 0);
  $("asr-rtf").textContent = number(asr.rtf, "×", 2);
  $("tts-rtf").textContent = number(tts.rtf, "×", 2);
  $("wer").textContent = number(
    data.quality?.asr?.wer == null ? null : data.quality.asr.wer * 100,
    "%",
  );
  $("resolved-device").textContent = data.device || "—";
  for (const id of ["latency", "ttft", "first-audio"])
    $(id).classList.toggle("missing", $(id).textContent === "—");
  renderStages(stages);
  if (data.system) renderSystem(data.system);
  if (data.detected_language)
    $("mic-state").textContent =
      data.detected_language === "ur" ? "Urdu detected" : "English detected";
  if (data.transcript && state.activeMessage)
    state.activeMessage.textContent = data.transcript;
  const nearBottom =
    $("messages").scrollHeight -
      $("messages").scrollTop -
      $("messages").clientHeight <
    80;
  if (data.reply && !state.interrupted) {
    if (!state.assistantMessage)
      state.assistantMessage = addMessage("assistant", data.reply);
    else state.assistantMessage.textContent = data.reply;
    if (nearBottom) $("messages").scrollTop = $("messages").scrollHeight;
  }
  for (const delta of data.deltas || []) {
    if (delta.seq <= state.lastDelta) continue;
    state.lastDelta = delta.seq;
    if (
      delta.kind === "audio_chunk" &&
      $("autoplay").checked &&
      !state.interrupted
    ) {
      playback.enqueue(delta.pcm, delta.sample_rate);
      state.streamedAudio = true;
    }
  }
}
async function responseJSON(response) {
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  return data;
}
function finish(data) {
  state.stream?.close();
  state.stream = null;
  state.busy = false;
  delete data.deltas;
  data.browser = {
    ...(state.browserMetrics || {}),
    request_to_result_s: (performance.now() - state.submittedAt) / 1000,
  };
  renderMetrics(data);
  if (data.operation === "warmup" && data.status === "ok") {
    state.prepared = true;
    state.preparing = false;
    notice(
      live.active
        ? "Ready and listening. Speak naturally."
        : "Models ready. Start conversation when ready.",
    );
    controls();
    return;
  }
  state.results.push(data);
  state.results = state.results.slice(-32);
  $("turn-count").textContent =
    `${state.results.length} ${state.results.length === 1 ? "turn" : "turns"}`;
  $("export").disabled = false;
  $("recent").replaceChildren(
    ...state.results
      .slice(-5)
      .reverse()
      .map((row, index) => {
        const element = document.createElement("div");
        element.className = `recent-row ${row.status}`;
        const label = document.createElement("span");
        label.textContent = `Turn ${state.results.length - index} · ${row.status === "error" ? "failed" : row.cold_in_process ? "model load" : "warm"}`;
        const value = document.createElement("span");
        value.textContent = secs(row.total_wall_s);
        element.append(label, value);
        return element;
      }),
  );
  $("reply-warning").hidden = !data.reply_truncated;
  if (data.status === "cancelled" && !state.bargeIn) {
    state.activeConversation = false;
    live.stop();
  }
  if (data.operation === "warmup" && data.status === "ok")
    state.prepared = true;
  state.preparing = false;
  if (data.status === "cancelled")
    notice("Reply interrupted. Ready for your next message.");
  else if (data.status === "error")
    notice(
      data.error || "Turn failed. See the server terminal for details.",
      true,
    );
  else {
    notice(
      data.reply_truncated
        ? "Reply reached its limit. Use Continue answer for the rest."
        : state.streamedAudio
          ? "Reply streamed. Audio may still be playing."
          : "Turn complete. Timings include preparation and model loading.",
    );
    if (data.audio_url) {
      $("player-wrap").hidden = false;
      $("player").src = data.audio_url;
      $("download").href = data.audio_url;
      $("player").onplaying = () => {
        if (data.browser.first_playback_s == null)
          data.browser.first_playback_s =
            (performance.now() - state.submittedAt) / 1000;
      };
      if ($("autoplay").checked && !state.streamedAudio && !state.interrupted)
        $("player")
          .play()
          .catch(() => notice("Reply ready. Press Play to hear it."));
    }
  }
  controls();
  maybeContinue();
}
async function submitTurn({ text = "", audio = null, prepare = false } = {}) {
  if (state.busy || state.recording || state.starting) return;
  const config = settings();
  if (!prepare && config.engine === "native" && $("mode").value === "speak") {
    notice(
      "Read-aloud voice preview uses the Python engine. Choose Python, or choose Reply with voice for native conversation.",
      true,
    );
    return;
  }
  const needsVoice = ["run", "speak"].includes($("mode").value);
  if (
    needsVoice &&
    config.tts_backend === "indic-mio" &&
    !$("voice-file").files.length
  ) {
    notice(
      "Indic-Mio needs a reference voice clip. Open Reference voice and choose a WAV or FLAC file.",
      true,
    );
    return;
  }
  if (
    needsVoice &&
    config.tts_backend === "omnivoice" &&
    $("voice-file").files.length &&
    !config.reference_text
  ) {
    notice("Add the exact words spoken in the reference clip.", true);
    return;
  }
  if (needsVoice && config.voice_description && $("voice-file").files.length) {
    notice(
      "Clear the voice description to use a reference voice, or clear the reference clip.",
      true,
    );
    return;
  }
  if (audio && $("mode").value === "speak") {
    notice(
      "Read my text aloud needs a typed message. Choose a reply or transcription mode for audio.",
      true,
    );
    return;
  }
  if (!prepare && !audio && $("mode").value === "transcribe") {
    notice("Record or upload audio for transcription.", true);
    return;
  }
  playback.stop();
  if ($("autoplay").checked) playback.prepare();
  state.lastDelta = 0;
  state.streamedAudio = false;
  state.interrupted = false;
  state.bargeIn = false;
  state.preparing = prepare;
  state.currentJob = null;
  state.browserMetrics = {};
  $("reply-warning").hidden = true;
  state.busy = true;
  controls();
  state.assistantMessage = null;
  state.submittedAt = performance.now();
  $("player").pause();
  $("player-wrap").hidden = true;
  state.activeMessage = prepare
    ? null
    : addMessage("user", text || "Audio recording · waiting for transcript…");
  notice(
    "Preparing turn. The first use may download model weights; live metrics will continue during loading.",
  );
  $("stages").replaceChildren();
  renderMetrics({ status: "running", phase: "Submitting", stages: {} });
  const form = new FormData();
  form.append("session", state.session);
  form.append("settings", JSON.stringify(config));
  form.append("mode", prepare ? "warmup" : $("mode").value);
  if (audio && $("expected-transcript").value.trim())
    form.append("expected_transcript", $("expected-transcript").value.trim());
  if (audio) form.append("audio", audio, "recording.wav");
  else form.append("text", text);
  if (
    needsVoice &&
    config.tts_backend !== "liquid" &&
    $("voice-file").files.length
  )
    form.append("voice", $("voice-file").files[0]);
  try {
    const result = await responseJSON(
      await fetch("/api/turn", { method: "POST", body: form }),
    );
    $("message").value = "";
    state.currentJob = result.id;
    if (state.interrupted) {
      await fetch(`/api/jobs/${result.id}/cancel?session=${state.session}`, {
        method: "POST",
      });
    }
    const url = `/api/jobs/${result.id}?session=${state.session}`;
    const source = new EventSource(
      `/api/jobs/${result.id}/events?session=${state.session}`,
    );
    state.stream = source;
    source.onmessage = (event) => {
      if (state.currentJob !== result.id) return;
      const data = JSON.parse(event.data);
      renderMetrics(data);
      if (data.status !== "running") finish(data);
    };
    source.onerror = async () => {
      if (state.currentJob !== result.id) return;
      source.close();
      state.stream = null;
      notice("Live connection interrupted. Checking the turn…");
      // Continue monitoring the same job; never silently start a duplicate inference.
      let failures = 0;
      while (state.busy && state.currentJob === result.id) {
        try {
          const data = await responseJSON(
            await fetch(url + `&after=${state.lastDelta}`),
          );
          failures = 0;
          renderMetrics(data);
          if (data.status !== "running") {
            finish(data);
            return;
          }
        } catch (error) {
          failures++;
          if (failures >= 5) {
            state.busy = false;
            state.preparing = false;
            await cancelReply(false);
            controls();
            notice(
              `Connection lost. The server may still be processing: ${error.message}`,
              true,
            );
            return;
          }
        }
        await new Promise((resolve) => setTimeout(resolve, 1500));
      }
    };
  } catch (error) {
    state.busy = false;
    state.preparing = false;
    await cancelReply(false);
    controls();
    notice(error.message, true);
  }
}
async function enumerateMics() {
  if (!navigator.mediaDevices) return;
  const previous = $("mic-device").value;
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(
    (item) => item.kind === "audioinput",
  );
  $("mic-device").replaceChildren(
    new Option("System default microphone", ""),
    ...devices.map(
      (item, index) =>
        new Option(item.label || `Microphone ${index + 1}`, item.deviceId),
    ),
  );
  if ([...$("mic-device").options].some((option) => option.value === previous))
    $("mic-device").value = previous;
}
function paintWave(samples = null) {
  const canvas = $("waveform"),
    ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth,
    height = 64;
  if (canvas.width !== Math.round(width * ratio))
    canvas.width = Math.round(width * ratio);
  if (canvas.height !== height * ratio) canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#e3e6ea";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
  if (!samples) return;
  ctx.strokeStyle = "#1d5fbf";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let x = 0; x < width; x++) {
    const value =
      samples[
        Math.min(samples.length - 1, Math.floor((x / width) * samples.length))
      ];
    const y = height / 2 - value * height * 0.45;
    if (x) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
  }
  ctx.stroke();
}
function wavBlob(chunks, sampleRate) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0),
    buffer = new ArrayBuffer(44 + length * 2),
    view = new DataView(buffer);
  const write = (at, text) => {
    for (let i = 0; i < text.length; i++)
      view.setUint8(at + i, text.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, length * 2, true);
  let offset = 44;
  for (const chunk of chunks)
    for (const sample of chunk) {
      const value = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, value < 0 ? value * 32768 : value * 32767, true);
      offset += 2;
    }
  return new Blob([buffer], { type: "audio/wav" });
}
async function startRecording(test = false) {
  if (state.recording || state.starting || state.busy) return;
  playback.stop();
  if ($("autoplay").checked) playback.prepare();
  state.starting = true;
  controls();
  const generation = ++state.startGeneration;
  const checkStart = () => {
    if (document.hidden || generation !== state.startGeneration)
      throw new DOMException(
        "Recording cancelled because the page was hidden.",
        "AbortError",
      );
  };
  let media = null,
    context = null;
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia)
      throw new Error(
        "Open http://127.0.0.1:8765 in Chrome or Safari. Microphone recording needs localhost or HTTPS.",
      );
    $("player").pause();
    notice("Allow microphone access in your browser.");
    $("mic-state").textContent = "Requesting access";
    const selected = $("mic-device").value;
    media = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: selected ? { exact: selected } : undefined,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
      video: false,
    });
    checkStart();
    context = new AudioContext();
    await context.resume();
    checkStart();
    await context.audioWorklet.addModule("/static/recorder-worklet.js");
    checkStart();
    const source = context.createMediaStreamSource(media),
      analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    const worklet = new AudioWorkletNode(context, "pcm-recorder");
    const silent = context.createGain();
    silent.gain.value = 0;
    source.connect(analyser);
    analyser.connect(worklet);
    worklet.connect(silent);
    silent.connect(context.destination);
    const recording = {
      test,
      media,
      context,
      source,
      analyser,
      worklet,
      silent,
      chunks: [],
      started: performance.now(),
      seenSpeech: false,
      lastSpeech: 0,
      peak: 0,
      stopping: false,
    };
    state.recording = recording;
    state.starting = false;
    worklet.port.onmessage = ({ data }) => {
      if (data === "flushed") recording.flushed?.();
      else if (data === "limit") stopRecording();
      else recording.chunks.push(data);
    };
    $("mic-state").textContent = test
      ? "Checking · no model loaded"
      : "Recording";
    enumerateMics().catch(() => {});
    controls();
    notice(
      test
        ? "Microphone check only. Speak and watch the input level."
        : "Recording. Press Stop & send when finished.",
    );
    const samples = new Float32Array(analyser.fftSize);
    const frame = () => {
      if (state.recording !== recording || recording.stopping) return;
      analyser.getFloatTimeDomainData(samples);
      paintWave(samples);
      const rms = Math.sqrt(
          samples.reduce((sum, value) => sum + value * value, 0) /
            samples.length,
        ),
        now = performance.now(),
        elapsed = (now - recording.started) / 1000;
      recording.peak = Math.max(recording.peak, rms);
      $("rms").textContent = rms.toFixed(5);
      $("record-time").textContent = `${elapsed.toFixed(1)} / 30 s`;
      if (rms >= 0.012) {
        recording.seenSpeech = true;
        recording.lastSpeech = now;
      }
      if (
        elapsed >= 29.5 ||
        (!test &&
          $("auto-stop").checked &&
          recording.seenSpeech &&
          now - recording.lastSpeech >= 900)
      ) {
        stopRecording();
        return;
      }
      recording.animation = requestAnimationFrame(frame);
    };
    frame();
    recording.timeout = setTimeout(() => stopRecording(), 29500);
  } catch (error) {
    media?.getTracks().forEach((track) => track.stop());
    await context?.close().catch(() => {});
    state.starting = false;
    state.recording = null;
    controls();
    $("mic-state").textContent = "Off";
    const message =
      error.name === "NotAllowedError"
        ? "Microphone permission was denied. Allow it in the browser site settings and macOS Privacy & Security → Microphone for this browser."
        : error.message;
    notice(message, error.name !== "AbortError");
  }
}
async function stopRecording(discard = false) {
  const recording = state.recording;
  if (recording && discard) recording.discard = true;
  if (!recording || recording.stopping) return;
  recording.stopping = true;
  cancelAnimationFrame(recording.animation);
  clearTimeout(recording.timeout);
  // Stop input first, then drain the worklet's partial buffer before encoding WAV.
  recording.source.disconnect();
  recording.media.getTracks().forEach((track) => track.stop());
  await Promise.race([
    new Promise((resolve) => {
      recording.flushed = resolve;
      recording.worklet.port.postMessage("flush");
    }),
    new Promise((resolve) => setTimeout(resolve, 500)),
  ]);
  recording.worklet.disconnect();
  recording.analyser.disconnect();
  recording.silent.disconnect();
  await recording.context.close().catch(() => {});
  state.recording = null;
  $("mic-state").textContent = "Off";
  controls();
  paintWave();
  if (discard || recording.discard) return;
  if (recording.test) {
    notice(
      `Microphone check: maximum RMS ${recording.peak.toFixed(5)}. ${recording.peak >= 0.012 ? "Input crossed the speech threshold." : "Low input. Check the selected device and input volume."}`,
    );
    return;
  }
  if (!recording.chunks.length || recording.peak < 0.00001) {
    state.activeConversation = false;
    notice(
      "The browser captured silence. Check the selected microphone and its permission.",
      true,
    );
    return;
  }
  if ($("auto-stop").checked && !recording.seenSpeech) {
    state.activeConversation = false;
    notice(
      "No speech crossed the microphone threshold. Start the conversation again when ready.",
      true,
    );
    controls();
    return;
  }
  await submitTurn({
    audio: wavBlob(recording.chunks, recording.context.sampleRate),
  });
}
$("record").onclick = () => {
  if (live.active) cancelReply(false);
  else if ($("continuous").checked) startLiveConversation();
  else if (state.recording) stopRecording();
  else {
    state.activeConversation = true;
    startRecording(false);
  }
};
$("check-mic").onclick = () =>
  state.recording ? stopRecording() : startRecording(true);
$("composer").onsubmit = (event) => {
  event.preventDefault();
  const text = $("message").value.trim();
  if (text) submitTurn({ text });
};
$("message").onkeydown = (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
};
$("audio-file").onchange = () => {
  const file = $("audio-file").files[0];
  if (file) {
    if (file.size > 12 * 1024 * 1024)
      notice(
        "Choose a WAV or FLAC file smaller than 12 MiB and no longer than 30 seconds.",
        true,
      );
    else submitTurn({ audio: file });
  }
  $("audio-file").value = "";
};
$("mode").onchange = configureVoice;
$("language").onchange = () => {
  configureVoice();
  controls();
};
$("tts").onchange = () => {
  configureVoice();
  controls();
};
$("engine").onchange = () => {
  if ($("engine").value === "bilingual") {
    $("language").value = "auto";
    $("tts").value = "omnivoice";
  } else if ($("engine").value === "python" && $("language").value === "auto")
    $("language").value = "ur";
  if ($("engine").value === "native") {
    $("language").value = "en";
    $("tts").value = "liquid";
  }
  configureVoice();
  controls();
};
$("warmup").onclick = () =>
  submitTurn({ text: "Prepare native model", prepare: true });
$("interrupt").onclick = () => cancelReply(false);
$("continue").onclick = () =>
  submitTurn({
    text: "Please continue the previous answer from where you stopped. Complete the remaining points in short, complete sentences.",
  });
$("autoplay").onchange = () => {
  if (!$("autoplay").checked) playback.stop();
};
$("speed").oninput = () =>
  ($("speed-label").textContent = `${Number($("speed").value).toFixed(1)}×`);
$("clear-voice").onclick = () => {
  $("voice-file").value = "";
  $("reference-text").value = "";
};
$("reset").onclick = () => {
  state.activeConversation = false;
  playback.stop();
  state.session = crypto.randomUUID();
  state.results = [];
  state.activeMessage = null;
  state.assistantMessage = null;
  $("messages").replaceChildren();
  $("turn-count").textContent = "0 turns";
  $("player").pause();
  $("player-wrap").hidden = true;
  $("export").disabled = true;
  $("recent").replaceChildren();
  $("stages").replaceChildren();
  renderMetrics({ phase: "Idle", stages: {} });
  notice("New conversation. Models stay loaded for the next turn.");
};
$("export").onclick = () => {
  const blob = new Blob([JSON.stringify(state.results, null, 2)], {
      type: "application/json",
    }),
    url = URL.createObjectURL(blob),
    link = document.createElement("a");
  link.href = url;
  link.download = "stts-session-metrics.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function refreshSystem() {
  if (state.stream) return;
  try {
    const system = await responseJSON(await fetch("/api/system"));
    renderSystem(system);
    $("native-ready").textContent =
      $("engine").value === "bilingual"
        ? system.bilingual_available
          ? "English/Urdu models are installed. Start conversation prepares them automatically."
          : "Bilingual models need setup. Run python setup_bilingual.py."
        : system.native_available
          ? "English model is installed. Start conversation prepares it automatically."
          : "Run python setup_native.py.";
    $("connection").textContent = "Connected";
    $("freshness").textContent = "Live";
  } catch {
    $("connection").textContent = "Disconnected";
    $("freshness").textContent = "Unavailable";
  }
}
window.addEventListener("pagehide", () => {
  live.stop();
  state.startGeneration++;
  stopRecording(true);
  state.stream?.close();
  playback.stop();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    if (live.active || state.starting) cancelReply(false);
    state.startGeneration++;
    if (state.recording) stopRecording(true);
    if (state.starting || state.recording)
      notice(
        "Recording cancelled because the page was hidden. Record again when ready.",
      );
  }
});
window.addEventListener("resize", () => {
  if (!state.recording) paintWave();
});
renderMetrics({ phase: "Idle", stages: {} });
configureVoice();
controls();
paintWave();
enumerateMics().catch(() => {});
refreshSystem();
setInterval(refreshSystem, 2000);
