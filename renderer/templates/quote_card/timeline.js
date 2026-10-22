// quote_card: card fades up, quote mark pops, attribution follows.
AR.register("quote_card", function (tl, el, shot) {
  const t = shot.start;
  const q = (s) => el.querySelector(s);
  tl.fromTo(q(".qc-card"), { opacity: 0, y: 40 }, { opacity: 1, y: 0, duration: 0.45, ease: "power2.out" }, t + 0.05);
  tl.fromTo(q(".qc-mark"), { scale: 0.4, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.4, ease: "back.out(2)" }, t + 0.25);
  if (q(".qc-attr")) tl.fromTo(q(".qc-attr"), { opacity: 0 }, { opacity: 0.92, duration: 0.3 }, t + 0.6);
});
