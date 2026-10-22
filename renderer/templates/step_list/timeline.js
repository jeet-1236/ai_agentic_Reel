// step_list: title wipes in, steps appear one by one over the first ~45% of the shot (max 0.5 s apart).
AR.register("step_list", function (tl, el, shot) {
  const t = shot.start, d = shot.end - shot.start;
  const title = el.querySelector(".st-title-row");
  if (title) tl.fromTo(title, { opacity: 0, x: -40 }, { opacity: 1, x: 0, duration: 0.35, ease: "power2.out" }, t + 0.05);
  const items = el.querySelectorAll(".st-item");
  const gap = Math.min(0.5, (d * 0.45) / Math.max(1, items.length));
  items.forEach((it, i) => {
    tl.fromTo(it, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.3, ease: "power2.out" }, t + 0.2 + gap * i);
  });
});
