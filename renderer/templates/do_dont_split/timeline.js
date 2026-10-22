// do_dont_split: panels slide in from opposite sides, items stagger in (all within ~1 s).
AR.register("do_dont_split", function (tl, el, shot, env) {
  const t = shot.start;
  const horiz = env.aspect === "16x9";
  const panels = el.querySelectorAll(".dd-panel");
  tl.fromTo(panels[0], horiz ? { opacity: 0, x: -60 } : { opacity: 0, y: -40 }, { opacity: 1, x: 0, y: 0, duration: 0.4, ease: "power2.out" }, t + 0.05);
  tl.fromTo(panels[1], horiz ? { opacity: 0, x: 60 } : { opacity: 0, y: 40 }, { opacity: 1, x: 0, y: 0, duration: 0.4, ease: "power2.out" }, t + 0.25);
  el.querySelectorAll(".dd-item").forEach((it, i) => {
    tl.fromTo(it, { opacity: 0, x: 24 }, { opacity: 1, x: 0, duration: 0.3, ease: "power2.out" }, t + 0.35 + 0.1 * i);
  });
});
