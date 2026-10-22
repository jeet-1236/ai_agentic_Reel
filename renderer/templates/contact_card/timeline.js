// contact_card: title drops in, each contact card slides up.
AR.register("contact_card", function (tl, el, shot) {
  const t = shot.start;
  const title = el.querySelector(".cc-title");
  if (title) tl.fromTo(title, { opacity: 0, y: -30 }, { opacity: 1, y: 0, duration: 0.35, ease: "power2.out" }, t + 0.05);
  el.querySelectorAll(".cc-item").forEach((it, i) => {
    tl.fromTo(it, { opacity: 0, y: 40 }, { opacity: 1, y: 0, duration: 0.35, ease: "power2.out" }, t + 0.25 + 0.15 * i);
  });
});
