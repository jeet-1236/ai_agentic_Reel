// end_card: logo fades in (never transformed beyond a uniform scale), CTA rises, URL + disclosure follow.
// Everything is fully visible from shot.start + 0.9 s so the critic's OCR of the last 4 s sees it.
AR.register("end_card", function (tl, el, shot) {
  const t = shot.start;
  const q = (s) => el.querySelector(s);
  tl.fromTo(q(".ec-logo-box"), { opacity: 0 }, { opacity: 1, duration: 0.35, ease: "power1.out" }, t + 0.05);
  tl.fromTo(q(".ec-cta"), { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.4, ease: "power2.out" }, t + 0.2);
  if (q(".ec-url")) tl.fromTo(q(".ec-url"), { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.35, ease: "power2.out" }, t + 0.4);
  tl.fromTo(q(".ec-disc"), { opacity: 0 }, { opacity: 1, duration: 0.3 }, t + 0.55);
});
