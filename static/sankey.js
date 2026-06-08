/* ── Insurance Premium Flow — Sankey renderer v5 ───────────────────────────
   Five-column layout:
     col0  Premium bar
     col1  Allocation after fees  (carrier grey + fee bands peeling off below)
     col2  Allocation after claims (claims / carrier profits / contingent profits)
     col3  Contingent split        (carrier contingent / MGA contingent) — only if >0
     col4  Final Split             (claims / carrier total / MGA total) — always shown

   Sliding profits: when enabled, MGA contingent = 50% of contingent profits.
   Because contingent profits shrink linearly to $0 as claims rise, MGA's
   absolute dollar amount also slides linearly down to $0 in the same fashion.

   Final Split bar height = total * sc, divided proportionally among the three
   buckets. When claims > paf (overrun) the three buckets are renormalised so
   the bar never overflows.
──────────────────────────────────────────────────────────────────────────── */
(function () {

  const C = {
    premium:          "#4A4A4A",
    carrier:          "#888780",
    claims:           "#A32D2D",
    carrierProfits:   "#0F6E56",
    contingent:       "#1A9E8A",
    carrierContingent:"#0D6E60",
    mgaContingent:    "#185FA5",
    mgaTotal:         "#7C5CBF",   // purple — MGA's aggregate bucket in Final Split
  };

  // ── Formatters ─────────────────────────────────────────────────────────────
  function fmt(v) {
    if (Math.abs(v) >= 1e6) return "$" + d3.format(",.2s")(v);
    return "$" + d3.format(",.0f")(v);
  }
  function fmtPct(v, total) {
    if (!total) return "—";
    return (v / total * 100).toFixed(0) + "%";
  }

  // ── Waterfall ribbon ───────────────────────────────────────────────────────
  function ribbon(x0, y0, h0, x1, y1, h1) {
    const mx = (x0 + x1) / 2;
    return [
      `M${x0},${y0}`,
      `C${mx},${y0} ${mx},${y1} ${x1},${y1}`,
      `L${x1},${y1 + h1}`,
      `C${mx},${y1 + h1} ${mx},${y0 + h0} ${x0},${y0 + h0}`,
      `Z`,
    ].join(" ");
  }

  // ── Main render ────────────────────────────────────────────────────────────
  window.renderSankey = function renderSankey(state) {
    const {
      total, fees, totalFees, premiumAfterFees,
      claimsActual, realisedLR, realisedProfitRatioPct,
      carrierProfits, contingentProfits,
      mgaContingentAmount, carrierContingentAmount,
      nodeWidth: NW,
    } = state;

    const hasContingent = contingentProfits > 0.005;

    const container = document.getElementById("chart-container");
    const W = container.clientWidth  || 900;
    const H = container.clientHeight || 560;

    const svg = d3.select("#sankey-svg").attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const defs = svg.append("defs");

    // ── Drop-shadow filter for superimposed labels ─────────────────────────
    const filt = defs.append("filter").attr("id", "tshadow")
      .attr("x", "-20%").attr("y", "-40%")
      .attr("width", "140%").attr("height", "180%");
    filt.append("feDropShadow")
      .attr("dx", 0).attr("dy", 0)
      .attr("stdDeviation", 2.5)
      .attr("flood-color", "#000")
      .attr("flood-opacity", 0.85);

    // ── Layout (5 columns always reserved) ───────────────────────────────
    const M = { top: 44, bottom: 36, left: 104, right: 80 };
    const iW = W - M.left - M.right;
    const iH = H - M.top  - M.bottom;
    const g  = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);

    const colGap = Math.min(140, Math.max(50, (iW - NW * 5) / 4));
    const col0 = 0;
    const col1 = col0 + NW + colGap;
    const col2 = col1 + NW + colGap;
    const col3 = col2 + NW + colGap;
    const col4 = col3 + NW + colGap;

    // Scale: claims can exceed total — fit whichever is taller
    const sc = iH / Math.max(total, claimsActual);

    // ── Stage 1: carrier (top) + fee bands (below) ────────────────────────
    const carrierH = premiumAfterFees * sc;

    const feeRects = [];
    let fTop = carrierH;
    fees.forEach((f, i) => {
      const h = f.value * sc;
      feeRects.push({ ...f, y: fTop, h, idx: i });
      fTop += h + (i < fees.length - 1 ? 2 : 0);
    });

    // ── Stage 2: carrier split ────────────────────────────────────────────
    const claimsH  = Math.max(0, claimsActual)     * sc;
    const profitsH = Math.max(0, carrierProfits)    * sc;
    const contH    = Math.max(0, contingentProfits) * sc;

    const claimsY  = 0;
    const profitsY = claimsH;
    const contY    = claimsH + profitsH;

    // When claims > paf the ribbon fans out from the full carrier band
    const claimsSrcH = Math.min(claimsH, carrierH);

    // ── Stage 3: contingent split ─────────────────────────────────────────
    const carrContH = Math.max(0, carrierContingentAmount) * sc;
    const mgaContH  = Math.max(0, mgaContingentAmount)     * sc;
    const carrContY = contY;
    const mgaContY  = carrContY + carrContH;

    // ── Stage 4: final split ──────────────────────────────────────────────
    const finalClaims  = claimsActual;
    const finalCarrier = carrierProfits + carrierContingentAmount;
    const finalMGA     = totalFees + mgaContingentAmount;
    const finalSum     = finalClaims + finalCarrier + finalMGA;
    const finalBarH    = total * sc;
    // Renormalise when overrun (finalSum > total) so bar never overflows
    const fsc          = finalSum > 0 ? finalBarH / finalSum : 0;
    const col4ClaimsH  = finalClaims  * fsc;
    const col4CarrierH = finalCarrier * fsc;
    const col4MGAH     = finalMGA     * fsc;
    const col4ClaimsY  = 0;
    const col4CarrierY = col4ClaimsH;
    const col4MGAY     = col4ClaimsH + col4CarrierH;

    // ── Gradients ─────────────────────────────────────────────────────────
    function addGrad(id, hex) {
      const gr = defs.append("linearGradient").attr("id", id)
        .attr("x1", "0%").attr("x2", "100%");
      gr.append("stop").attr("offset", "0%")
        .attr("stop-color", hex).attr("stop-opacity", 0.94);
      gr.append("stop").attr("offset", "100%")
        .attr("stop-color", hex).attr("stop-opacity", 0.76);
    }
    addGrad("gPrem",   C.premium);
    addGrad("gCarr",   C.carrier);
    addGrad("gClaim",  C.claims);
    addGrad("gProfit", C.carrierProfits);
    addGrad("gCont",   C.contingent);
    addGrad("gCCont",  C.carrierContingent);
    addGrad("gMCont",  C.mgaContingent);
    addGrad("gMGATot", C.mgaTotal);
    fees.forEach((f, i) => addGrad(`gFee${i}`, f.color));

    // ── Layers (z-order: ribbons → bars → labels) ─────────────────────────
    const ribbonG = g.append("g").attr("class", "ribbons");
    const nodeG   = g.append("g").attr("class", "nodes");
    const labelG  = g.append("g").attr("class", "labels");

    // ── Draw rect ─────────────────────────────────────────────────────────
    function drawRect(x, y, h, fill) {
      if (h < 0.5) return;
      nodeG.append("rect")
        .attr("x", x).attr("y", y)
        .attr("width", NW).attr("height", h)
        .attr("fill", fill).attr("rx", 2);
    }

    // ── Draw ribbon ───────────────────────────────────────────────────────
    function drawRibbon(x0, y0, h0, x1, y1, h1, grad) {
      if (h0 < 0.5 || h1 < 0.5) return;
      ribbonG.append("path").attr("class", "ribbon")
        .attr("d", ribbon(x0, y0, h0, x1, y1, h1))
        .attr("fill", `url(#${grad})`).attr("opacity", 0.36);
    }

    // ── Superimposed bar label (white, drop-shadow, clipped) ───────────────
    const LINE_H   = 14;
    const MIN_SHOW = 8;
    let   clipSeq  = 0;

    function lblBar(colX, barY, barH, ...lines) {
      if (barH < MIN_SHOW || !lines.length) return;

      const cid = `clip-bar-${clipSeq++}`;
      defs.append("clipPath").attr("id", cid)
        .append("rect")
          .attr("x", colX).attr("y", barY)
          .attr("width", NW).attr("height", barH);

      const clpG  = labelG.append("g").attr("clip-path", `url(#${cid})`);
      const maxFit = Math.max(1, Math.floor(barH / LINE_H));
      const visible = lines.slice(0, maxFit);
      const n      = visible.length;
      const cx     = colX + NW / 2;
      const firstY = barY + barH / 2 - (n - 1) * LINE_H / 2;

      visible.forEach((line, i) => {
        clpG.append("text")
          .attr("class", i === 0 ? "lbl-bar-primary" : "lbl-bar-secondary")
          .attr("text-anchor", "middle")
          .attr("x", cx)
          .attr("y", firstY + i * LINE_H)
          .attr("dy", "0.35em")
          .attr("filter", "url(#tshadow)")
          .text(line);
      });
    }

    // Left-of-bar label (col0 premium bar)
    const MIN_2LINE = 26;
    function lblLeft(barX, barY, barH, line1, line2) {
      if (barH < MIN_SHOW) return;
      const x = barX - 10;
      const y = barY + barH / 2;
      const grp = labelG.append("g").attr("transform", `translate(${x},${y})`);
      const showTwo = barH >= MIN_2LINE && line2;
      grp.append("text").attr("class", "lbl-primary").attr("text-anchor", "end")
        .attr("dy", showTwo ? "-0.25em" : "0.35em").text(line1);
      if (showTwo) {
        grp.append("text").attr("class", "lbl-secondary").attr("text-anchor", "end")
          .attr("dy", "1.1em").text(line2);
      }
    }

    // ── Col 0→1 ribbons ────────────────────────────────────────────────────
    drawRibbon(col0 + NW, 0, carrierH, col1, 0, carrierH, "gCarr");

    let srcFeeY = carrierH;
    feeRects.forEach((fr, i) => {
      drawRibbon(col0 + NW, srcFeeY, fr.h, col1, fr.y, fr.h, `gFee${fr.idx}`);
      srcFeeY += fr.h + (i < feeRects.length - 1 ? 2 : 0);
    });

    // ── Col 1→2 ribbons ────────────────────────────────────────────────────
    drawRibbon(col1 + NW, claimsY,  claimsSrcH, col2, claimsY,  claimsH,  "gClaim");
    drawRibbon(col1 + NW, profitsY, profitsH,   col2, profitsY, profitsH,  "gProfit");
    if (hasContingent) {
      drawRibbon(col1 + NW, contY, contH, col2, contY, contH, "gCont");
    }

    // ── Col 2→3 ribbons ────────────────────────────────────────────────────
    if (hasContingent) {
      drawRibbon(col2 + NW, carrContY, carrContH, col3, carrContY, carrContH, "gCCont");
      drawRibbon(col2 + NW, mgaContY,  mgaContH,  col3, mgaContY,  mgaContH,  "gMCont");
    }

    // ── Bars ──────────────────────────────────────────────────────────────
    drawRect(col0, 0, total * sc, C.premium);
    drawRect(col1, 0, carrierH,   C.carrier);
    feeRects.forEach(fr => drawRect(col1, fr.y, fr.h, fr.color));
    drawRect(col2, claimsY,  claimsH,  C.claims);
    drawRect(col2, profitsY, profitsH, C.carrierProfits);
    if (hasContingent) {
      drawRect(col2, contY,     contH,     C.contingent);
      drawRect(col3, carrContY, carrContH, C.carrierContingent);
      drawRect(col3, mgaContY,  mgaContH,  C.mgaContingent);
    }

    // col4 Final Split — separator then bars
    g.append("line")
      .attr("x1", col4 - colGap * 0.45).attr("x2", col4 - colGap * 0.45)
      .attr("y1", -M.top + 8).attr("y2", iH + M.bottom - 8)
      .attr("stroke", "#ddd").attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,3").attr("opacity", 0.7);

    drawRect(col4, col4ClaimsY,  col4ClaimsH,  C.claims);
    drawRect(col4, col4CarrierY, col4CarrierH, C.carrierProfits);
    drawRect(col4, col4MGAY,     col4MGAH,     C.mgaTotal);

    // ── Column headers ────────────────────────────────────────────────────
    const headers = [
      { x: col0 + NW / 2, t: "Premium" },
      { x: col1 + NW / 2, t: "Allocation after fees" },
      { x: col2 + NW / 2, t: "Allocation after claims" },
    ];
    if (hasContingent) headers.push({ x: col3 + NW / 2, t: "Contingent split" });
    headers.push({ x: col4 + NW / 2, t: "Final Split" });

    headers.forEach(({ x, t }) =>
      g.append("text").attr("class", "col-header")
        .attr("x", x).attr("y", -20)
        .attr("text-anchor", "middle").text(t));

    // ── Col 0: premium label (left of bar) ────────────────────────────────
    lblLeft(col0, 0, total * sc, fmt(total), "Total premium");

    // ── Col 1: carrier bar label ──────────────────────────────────────────
    lblBar(col1, 0, carrierH,
      fmt(premiumAfterFees),
      "Premium after fees");

    // ── Col 1: fee band labels (external, push-apart) ─────────────────────
    let lblPos = feeRects.map(fr => ({ fr, ly: fr.y + fr.h / 2 }));
    const MIN_GAP = 17;
    for (let pass = 0; pass < 12; pass++) {
      for (let i = 1; i < lblPos.length; i++) {
        const gap = lblPos[i].ly - lblPos[i - 1].ly;
        if (gap < MIN_GAP) {
          const push = (MIN_GAP - gap) / 2;
          lblPos[i - 1].ly -= push;
          lblPos[i].ly     += push;
        }
      }
    }

    lblPos.forEach(({ fr, ly }) => {
      if (fr.h < 2) return;
      const barMid = fr.y + fr.h / 2;
      const rx = col1 + NW;
      const needsLeader = Math.abs(ly - barMid) > 3 || fr.h < MIN_2LINE;
      const labelX = rx + (needsLeader ? 22 : 10);

      if (needsLeader) {
        const ex = rx + 8;
        labelG.append("polyline")
          .attr("points", `${rx},${barMid} ${ex},${barMid} ${ex},${ly} ${ex + 10},${ly}`)
          .attr("stroke", fr.color).attr("stroke-width", 1)
          .attr("fill", "none").attr("opacity", 0.65);
      }
      const wrap = labelG.append("g").attr("transform", `translate(${labelX},${ly})`);
      wrap.append("text").attr("class", "lbl-primary").attr("text-anchor", "start")
        .attr("dy", "-0.2em").text(fmt(fr.value));
      wrap.append("text").attr("class", "lbl-secondary").attr("text-anchor", "start")
        .attr("dy", "1em").text(fr.label);
    });

    // ── Col 2: superimposed labels ────────────────────────────────────────
    lblBar(col2, claimsY, claimsH,
      fmt(claimsActual),
      "Claims paid",
      `${(realisedLR * 100).toFixed(0)}% realised LR`);

    lblBar(col2, profitsY, profitsH,
      fmt(carrierProfits),
      "Carrier profits",
      `${(+realisedProfitRatioPct).toFixed(0)}% profit ratio`);

    if (hasContingent) {
      lblBar(col2, contY, contH,
        fmt(contingentProfits),
        "Contingent profits",
        `${fmtPct(contingentProfits, premiumAfterFees)} of PAF`);
    }

    // ── Col 3: superimposed labels ────────────────────────────────────────
    if (hasContingent) {
      lblBar(col3, carrContY, carrContH,
        fmt(carrierContingentAmount),
        "Carrier contingent");
      lblBar(col3, mgaContY, mgaContH,
        fmt(mgaContingentAmount),
        "MGA contingent");
    }

    // ── Col 4: Final Split labels ─────────────────────────────────────────
    lblBar(col4, col4ClaimsY, col4ClaimsH,
      fmt(finalClaims),
      "Claims",
      `${fmtPct(finalClaims, total)} of premium`);

    lblBar(col4, col4CarrierY, col4CarrierH,
      fmt(finalCarrier),
      "Carrier",
      `${fmtPct(finalCarrier, total)} of premium`);

    lblBar(col4, col4MGAY, col4MGAH,
      fmt(finalMGA),
      "MGA",
      `${fmtPct(finalMGA, total)} of premium`);

    // ── Originator fees bracket ───────────────────────────────────────────
    if (feeRects.length > 0 && totalFees > 0) {
      const fy1 = feeRects[0].y;
      const fy2 = feeRects[feeRects.length - 1].y + feeRects[feeRects.length - 1].h;
      if (fy2 - fy1 > 6) {
        const bx = col1 + NW + Math.min(colGap * 0.72, colGap - 28);
        const my = (fy1 + fy2) / 2;
        const bg = g.append("g").attr("class", "bracket-group");
        bg.append("line")
          .attr("x1", bx).attr("y1", fy1).attr("x2", bx).attr("y2", fy2)
          .attr("stroke", "#7C5CBF").attr("stroke-width", 1.5).attr("opacity", 0.45);
        [fy1, fy2].forEach(ty => bg.append("line")
          .attr("x1", bx - 5).attr("y1", ty).attr("x2", bx).attr("y2", ty)
          .attr("stroke", "#7C5CBF").attr("stroke-width", 1.5).attr("opacity", 0.45));
        bg.append("text").attr("class", "bracket-val").attr("fill", "#7C5CBF")
          .attr("x", bx + 7).attr("y", my - 4).text(fmt(totalFees));
        bg.append("text").attr("class", "bracket-label")
          .attr("x", bx + 7).attr("y", my + 11)
          .text(`${fmtPct(totalFees, total)} originator fees`);
      }
    }
  };

})();
