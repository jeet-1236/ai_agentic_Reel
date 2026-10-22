// stat_callout: the number pops in with a small overshoot, label and source follow.
AR.register("stat_callout", function (tl, el, shot) {
  const t = shot.start;
  const q = (s) => el.querySelector(s);
  tl.fromTo(q(".sc-stat"), { opacity: 0, scale: 0.6 }, { opacity: 1, scale: 1, duration: 0.5, ease: "back.out(1.7)" }, t + 0.05);
  if (q(".sc-label")) tl.fromTo(q(".sc-label"), { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.4, ease: "power2.out" }, t + 0.4);
  if (q(".sc-source")) tl.fromTo(q(".sc-source"), { opacity: 0 }, { opacity: 0.9, duration: 0.3 }, t + 0.7);
});
