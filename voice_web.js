// Browser speech backend for the Godot web export: vosk-browser (Kaldi compiled to wasm) in a
// worker, microphone captured here and fed straight to the recognizer. Godot talks to it through
// JavaScriptBridge (see project/scripts/voice/web_recognizer.gd).
(function () {
  const S = { model: null, rec: null, ctx: null, stream: null, node: null, src: null, sink: null, cb: null, vocab: "[]", starting: null };
  const BATCH = 2048;

  async function load(url, vocabJson, onReady, onPartial, onFinal, onLevel, onError) {
    S.cb = { onReady, onPartial, onFinal, onLevel, onError };
    S.vocab = JSON.stringify(JSON.parse(vocabJson).concat(["[unk]"]));
    try {
      S.model = await Vosk.createModel(new URL(url, window.location.href).href);
      console.log("[voice] model ready");
      onReady();
    } catch (e) {
      onError("model: " + e);
    }
  }

  function startMic() {
    if (S.node) return;
    if (S.starting) return;
    S.starting = (async () => {
      try {
        if (!S.model) throw new Error("model not loaded yet");
        S.stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
        });
        S.ctx = new AudioContext();
        S.ctx.resume().catch(() => {});
        const unlock = () => { if (S.ctx) S.ctx.resume().catch(() => {}); };
        for (const ev of ["pointerdown", "keydown"]) window.addEventListener(ev, unlock, { once: true });

        const code = "class Cap extends AudioWorkletProcessor{process(i){const c=i[0][0];if(c)this.port.postMessage(c.slice(0));return true;}}registerProcessor('cap',Cap);";
        await S.ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([code], { type: "application/javascript" })));

        const rate = S.ctx.sampleRate;
        S.rec = new S.model.KaldiRecognizer(rate, S.vocab);
        S.rec.setWords(true);
        S.rec.on("partialresult", (m) => S.cb.onPartial(m.result.partial || ""));
        S.rec.on("result", (m) => {
          const r = m.result || {};
          if (!r.text) return;
          const words = (r.result || []).map((w) => ({ word: w.word, conf: w.conf }));
          console.log("[voice] final:", r.text);
          S.cb.onFinal(JSON.stringify({ text: r.text, words: words }));
        });

        let buf = new Float32Array(0);
        S.node = new AudioWorkletNode(S.ctx, "cap");
        S.node.port.onmessage = (e) => {
          const chunk = e.data;
          const merged = new Float32Array(buf.length + chunk.length);
          merged.set(buf);
          merged.set(chunk, buf.length);
          buf = merged;
          if (buf.length < BATCH) return;
          let peak = 0;
          for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i]));
          S.cb.onLevel(peak);
          if (S.rec) S.rec.acceptWaveformFloat(buf, rate);
          buf = new Float32Array(0);
        };
        S.src = S.ctx.createMediaStreamSource(S.stream);
        S.src.connect(S.node);
        S.sink = S.ctx.createGain(); // keeps the worklet pulled without playing the mic back
        S.sink.gain.value = 0;
        S.node.connect(S.sink);
        S.sink.connect(S.ctx.destination);
      } catch (e) {
        stopMic();
        S.cb.onError("microphone: " + e);
      } finally {
        S.starting = null;
      }
    })();
  }

  function stopMic() {
    try { if (S.node) { S.node.port.onmessage = null; S.node.disconnect(); } } catch (e) {}
    try { if (S.src) S.src.disconnect(); } catch (e) {}
    try { if (S.rec) S.rec.remove(); } catch (e) {}
    if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
    if (S.ctx) S.ctx.close().catch(() => {});
    S.node = S.src = S.rec = S.stream = S.ctx = S.sink = null;
  }

  window.voiceWeb = { load, startMic, stopMic };
})();
