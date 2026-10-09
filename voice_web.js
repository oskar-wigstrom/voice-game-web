// Browser speech backend for the Godot web export: vosk-browser (Kaldi compiled to wasm) in a
// worker, microphone captured here and fed straight to the recognizer. Godot talks to it through
// JavaScriptBridge (see project/scripts/voice/web_recognizer.gd).
(function () {
  const S = { model: null, rec: null, ctx: null, stream: null, node: null, src: null, sink: null, cb: null, vocab: "[]", starting: null };
  const BATCH = 2048;
  // Energy gate (Schmitt trigger on a dB scale): audio only reaches the recognizer while someone is
  // talking. The noise floor is tracked while closed; opening needs OPEN_DB above it, staying open
  // needs CLOSE_DB above it, and it holds HOLD_S after the last loud batch so Kaldi sees enough
  // trailing silence (>= 1 s) to endpoint the utterance. PREROLL_S of audio before opening is replayed.
  const OPEN_DB = 12, CLOSE_DB = 6, MIN_OPEN_DBFS = -50, HOLD_S = 1.2, PREROLL_S = 0.4;
  const stats = { fedS: 0, totalS: 0 };

  function makeGate(rate) {
    const g = { open: false, floor: -70, quietS: 0, pre: [] };
    const dur = BATCH / rate;
    // returns the batches to feed (possibly empty) and whether the gate just closed
    g.push = (buf) => {
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const db = 10 * Math.log10(sum / buf.length + 1e-12);
      const openAt = Math.max(g.floor + OPEN_DB, MIN_OPEN_DBFS);
      const stayAt = Math.max(g.floor + CLOSE_DB, MIN_OPEN_DBFS - 6);
      const out = { feed: [], closed: false };
      stats.totalS += dur;
      if (!g.open) {
        g.floor += (db - g.floor) * (db < g.floor ? 0.3 : 0.05);
        if (db >= openAt) {
          g.open = true;
          g.quietS = 0;
          out.feed = g.pre.concat([buf]);
          g.pre = [];
        } else {
          g.pre.push(buf);
          while (g.pre.length * dur > PREROLL_S) g.pre.shift();
        }
      } else {
        out.feed = [buf];
        g.quietS = db >= stayAt ? 0 : g.quietS + dur;
        if (g.quietS >= HOLD_S) {
          g.open = false;
          out.closed = true;
        }
      }
      stats.fedS += out.feed.length * dur;
      return out;
    };
    return g;
  }

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
        const gate = makeGate(rate);
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
          const r = gate.push(buf);
          buf = new Float32Array(0);
          if (!S.rec) return;
          for (const b of r.feed) S.rec.acceptWaveformFloat(b, rate);
          if (r.closed) S.rec.retrieveFinalResult();
        };
        S.src = S.ctx.createMediaStreamSource(S.stream);
        S.src.connect(S.node);
        S.sink = S.ctx.createGain(); // keeps the worklet pulled without playing the mic back
        S.sink.gain.value = 0;
        S.node.connect(S.sink);
        S.sink.connect(S.ctx.destination);
      } catch (e) {
        teardown();
        S.cb.onError("microphone: " + e);
      } finally {
        S.starting = null;
      }
    })();
  }

  function stopMic() {
    S.resumeOnShow = false;
    teardown();
  }

  // Release the mic and recognizer while the tab is hidden; reopen when it is visible again.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (S.node) {
        S.resumeOnShow = true;
        teardown();
      }
    } else if (S.resumeOnShow) {
      S.resumeOnShow = false;
      startMic();
    }
  });

  function teardown() {
    try { if (S.node) { S.node.port.onmessage = null; S.node.disconnect(); } } catch (e) {}
    try { if (S.src) S.src.disconnect(); } catch (e) {}
    try { if (S.rec) S.rec.remove(); } catch (e) {}
    if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
    if (S.ctx) S.ctx.close().catch(() => {});
    S.node = S.src = S.rec = S.stream = S.ctx = S.sink = null;
  }

  window.voiceWeb = { load, startMic, stopMic, stats };
})();
