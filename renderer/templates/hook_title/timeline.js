// hook_title: kicker slides in, title rises, accent bar grows, subtitle fades in. All within 1 s of shot start.
AR.register("hook_title", function (tl, el, shot) {
  const t = shot.start;
  const q = (s) => el.querySelector(s);
  if (q(".kicker")) tl.fromTo(q(".kicker"), { opacity: 0, x: -40 }, { opacity: 1, x: 0, duration: 0.35, ease: "power2.out" }, t + 0.05);
  tl.fromTo(q(".title"), { opacity: 0, y: 60, scale: 0.96 }, { opacity: 1, y: 0, scale: 1, duration: 0.5, ease: "power3.out" }, t + 0.1);
  tl.fromTo(q(".bar"), { scaleX: 0 }, { scaleX: 1, duration: 0.4, ease: "power2.out" }, t + 0.4);
  if (q(".subtitle")) tl.fromTo(q(".subtitle"), { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.4, ease: "power2.out" }, t + 0.55);
  // slow push on the title for the rest of the shot (keeps it alive, stays inside the zone)
  tl.to(q(".title"), { scale: 1.03, duration: Math.max(0.1, shot.end - shot.start - 0.6), ease: "none" }, t + 0.6);
});
