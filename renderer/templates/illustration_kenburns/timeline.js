// illustration_kenburns: the Ken Burns move is applied by the driver to every shot image; here only the headline.
AR.register("illustration_kenburns", function (tl, el, shot) {
  const p = el.querySelector(".panel");
  if (p) tl.fromTo(p, { opacity: 0, y: -30 }, { opacity: 1, y: 0, duration: 0.45, ease: "power2.out" }, shot.start + 0.15);
});
