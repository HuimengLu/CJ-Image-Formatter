"use client";

import { Fragment, useEffect, useRef, useState } from "react";
import CompareSlider from "@/components/CompareSlider";
import DimensionOverlay from "@/components/DimensionOverlay";
import { ArrowClockwiseIcon, CameraPlusIcon, CheckCircleIcon, CircleIcon } from "@/components/icons";
import OptionCard from "@/components/OptionCard";
import {
  RATIOS, RATIO_AR, SESSION_EXPIRED, entryKey, imgUrl, stageW, thumbUrl, useListing,
  type Photo,
} from "./listing-state";

/* New Listing — the primary photo-processing pipeline: gpt-image-2 redraws the
   product on a white background with a studio shadow, multiply-blended onto the
   CJ backdrop. (The pre-AI local pipeline lives on at /legacy.) The backend
   routes keep their historical /api/testing2 prefix.

   Cover mode: Generate Cover classifies the product (gpt-5.4) and
   composes it into that category's scene (gpt-image-2 + static/cover bg). The
   scene lands as an EXTRA filmstrip entry (variant "cover") sharing the same
   backend id as the white-plate entry; its "before" is the original photo.

   All processing state lives in listing-state.tsx (a layout-level provider),
   so batches keep running and results survive tab switches. This component
   is the view: DOM refs and hover/drag visuals only. */

/* Filmstrip thumb: flat gray with a tiny "cooking…" until its image has
   actually downloaded (tracked via an off-DOM preload of the same URL). */
function ThumbButton({ p, active, onClick }: {
  p: Photo; active: boolean; onClick: () => void;
}) {
  const url = thumbUrl(p);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    setLoaded(false);
    const img = new window.Image();
    img.onload = () => setLoaded(true);
    img.src = url;
    return () => { img.onload = null; };
  }, [url]);
  return (
    <button
      className={`cj-thumb${active ? " active" : ""}`}
      style={loaded ? { backgroundImage: `url(${url})` } : undefined}
      onClick={onClick}
      aria-label={p.name}
    >
      {!loaded && <span className="cj-thumb-cooking">cooking…</span>}
    </button>
  );
}

function StepsStrip({ current = 0 }: { current?: number }) {
  const steps = ["Upload", "Optimize", "Export"];
  return (
    <div className="cj-strip">
      {steps.map((label, i) => (
        <span key={label} style={{ display: "contents" }}>
          {i > 0 && <span className="cj-dash">—</span>}
          <div className={`cj-step${current === i + 1 ? " active" : ""}`}>
            <span className="cj-num">{i + 1}</span>
            <span className="cj-steplbl">{label}</span>
          </div>
        </span>
      ))}
    </div>
  );
}

export default function ListingPage() {
  const {
    phase, rows, photos, active, warn, animate, imgLoading, pending,
    confirmDelete, exportOpen, exportSel, exporting, dimMode, coverBusy,
    failed, savedNote, newCovers, undoDelete, photo, donePct, sliderCtl,
    setPhotos, setActive, setWarn, setAnimate, setImgLoading, setDimMode,
    setConfirmDelete, setExportOpen, setExportSel,
    startBatch, cancelBatch, addMore, retryFailed, toggleCover,
    deletePhoto, undoRemove, exportSelected, exportSingle, mutatePhoto,
  } = useListing();

  const [dragover, setDragover] = useState(false);
  const [dropHover, setDropHover] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const moreRef = useRef<HTMLInputElement>(null);

  const multi = photos.length > 1;
  const coverOn = !!photo && photos.some((p) => p.id === photo.id && p.variant === "cover");
  const busy = !!photo && coverBusy.has(photo.id);

  /* ═══ home ═══ */
  if (phase === "home") {
    return (
      <div className="cj-home">
        <div className="cj-home-top" />
        {warn && <div className="cj-warn">&#9888; {warn}</div>}
        <div
          className={`cj-drop${dragover ? " dragover" : ""}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
          onDragLeave={() => setDragover(false)}
          onDrop={(e) => { e.preventDefault(); setDragover(false); startBatch(Array.from(e.dataTransfer.files)); }}
          onMouseEnter={() => setDropHover(true)}
          onMouseLeave={() => setDropHover(false)}
        >
          <div className="cj-drop-icon"><CameraPlusIcon size={40} /></div>
          <div className="cj-drop-title">Click or drop files</div>
          <div className="cj-drop-hint">JPG · PNG · WEBP / MAX 200MB</div>
          <input
            ref={fileRef} type="file" multiple hidden
            accept="image/jpeg,image/png,image/webp"
            onChange={(e) => { startBatch(Array.from(e.target.files ?? [])); e.target.value = ""; }}
          />
        </div>
        <div className="cj-drop-note">Got multiple photos? Upload them all. We&apos;ll process 3 at a time.</div>
        <StepsStrip current={dragover || dropHover ? 1 : 0} />
      </div>
    );
  }

  /* ═══ processing ═══ */
  if (phase === "processing") {
    return (
      <div className="cj-home">
        <div className="cj-home-top" />
        <div className="cj-proc">
          {rows.length > 1 && (
            <div>
              <div className="cj-proc-head">
                <span className="cj-proc-head-label">Batch Progress</span>
                <span className="cj-proc-head-pct">{donePct}%</span>
              </div>
              <div className="cj-proc-head-track">
                <div className="cj-proc-head-fill" style={{ width: `${donePct}%` }} />
              </div>
            </div>
          )}
          <div className="cj-proc-list">
            {rows.map((r, i) => (
              <div key={i} className={`cj-proc-row ${r.state}`}>
                {r.thumb
                  ? /* eslint-disable-next-line @next/next/no-img-element */
                    <img className="cj-proc-thumb" src={r.thumb} alt="" />
                  : <div className="cj-proc-thumb" style={{ background: "var(--line-soft)" }} />}
                <div className="cj-proc-body">
                  <div className="cj-proc-top">
                    <span className="cj-proc-name">{r.name}</span>
                    <span className="cj-proc-status">
                      {{ waiting: "Waiting…", active: "Optimizing… usually 20–40s", done: "Done", error: "Failed" }[r.state]}
                    </span>
                  </div>
                  <div className="cj-proc-track"><div className="cj-proc-fill" /></div>
                </div>
              </div>
            ))}
          </div>
          <button className="cj-proc-cancel" onClick={cancelBatch}>
            Cancel
          </button>
        </div>
        <StepsStrip current={2} />
      </div>
    );
  }

  /* ═══ result ═══ */
  if (!photo) return null;

  return (
    <div className="cj-result">
      {/* Global notices float over the stage — panel-buried warnings get missed. */}
      {warn && <div className="cj-warn cj-warn-float">&#9888; {warn}</div>}
      {undoDelete && (
        <div className="cj-undo">
          <span>Photo removed</span>
          <button onClick={undoRemove}>Undo</button>
        </div>
      )}
      <div className="cj-canvas-col">
        {/* is-loading lets mobile CSS hold the stage at the finished frame's
            aspect ratio while pixels are still downloading. */}
        <div className={`cj-stage${imgLoading ? " is-loading" : ""}`}>
          <CompareSlider
            key={entryKey(photo)}
            beforeSrc={imgUrl(photo, "before", stageW())}
            afterSrc={imgUrl(photo, "after", stageW())}
            aspectRatio={RATIO_AR[photo.ratio]}
            animate={animate}
            restX={0}
            onAfterLoaded={() => { setAnimate(false); setImgLoading(false); }}
            onAfterError={() => {
              setImgLoading(false);
              setWarn(SESSION_EXPIRED);
            }}
            controlRef={sliderCtl}
            overlay={
              (dimMode || photo.dimensions.length > 0) && (
                <DimensionOverlay
                  key={entryKey(photo)}
                  adding={dimMode}
                  dimensions={photo.dimensions}
                  onChange={(ds) => setPhotos((ps) => ps.map((p, i) => (i === active ? { ...p, dimensions: ds } : p)))}
                  onExitAdding={() => setDimMode(false)}
                  onInteract={() => sliderCtl.current?.toLeft()}
                />
              )
            }
          />
          {/* Single loading signal: the pac-man loader over the flat
              is-loading base (which also holds the finished frame's size). */}
          {imgLoading && <div className="cj-stage-loading"><div className="cj-pacman" /></div>}
        </div>

        {/* Always present — the "+" tile is the only way to add more photos. */}
        {(
          <div className="cj-film">
            {photos.map((p, i) => (
              <Fragment key={entryKey(p)}>
                <div className="cj-tw">
                  <ThumbButton
                    p={p}
                    active={i === active}
                    onClick={() => { if (i !== active) { setActive(i); setImgLoading(true); setDimMode(false); setAnimate(true); } }}
                  />
                  <button className="cj-delbtn" onClick={() => setConfirmDelete(i)} aria-label={`Delete ${p.name}`}>
                    ✕
                  </button>
                  {p.variant === "cover" && newCovers.has(p.id) && (
                    <span className="cj-newdot" title="New cover — click to view" />
                  )}
                </div>
                {/* A cover being generated shows up immediately as a loading
                    tile where the finished cover will land (right after its
                    base photo), styled like an uploading photo. The tile's
                    image is the cover's "before" (the original upload). */}
                {!p.variant && coverBusy.has(p.id) &&
                  !photos.some((q) => q.id === p.id && q.variant === "cover") && (
                    <div
                      className="cj-pend spin"
                      style={{ backgroundImage: `url(/api/testing2/${p.id}/image?kind=before&ratio=1:1&w=160)` }}
                      title="Generating cover…"
                      aria-label="Generating cover"
                    />
                  )}
              </Fragment>
            ))}
            {pending.map((t, i) => (
              <div
                key={`pend-${i}`}
                className={`cj-pend${t.state === "active" ? " spin" : t.state === "waiting" ? " dim" : ""}`}
                style={t.thumb ? { backgroundImage: `url(${t.thumb})` } : undefined}
              />
            ))}
            {failed.map((f, i) => (
              <button
                key={`fail-${i}`}
                className={`cj-fail${f.busy ? " busy" : ""}`}
                style={f.thumb ? { backgroundImage: `url(${f.thumb})` } : undefined}
                title={`${f.name} failed — click to retry${f.warn ? `\n${f.warn}` : ""}`}
                onClick={() => retryFailed(i)}
              >
                {!f.busy && <ArrowClockwiseIcon size={22} />}
              </button>
            ))}
            <button className="cj-addtile" onClick={() => moreRef.current?.click()} aria-label="Add photos">
              +
            </button>
          </div>
        )}
        <input
          ref={moreRef} type="file" multiple hidden
          accept="image/jpeg,image/png,image/webp"
          onChange={(e) => { addMore(Array.from(e.target.files ?? [])); e.target.value = ""; }}
        />
      </div>

      <aside className="cj-panel">
        <div className="cj-panel-body">
          <div className="cj-grp">
            <div className="cj-field">
              <span className="cj-label">Output Ratio</span>
              <div className="cj-ratio-row">
                {RATIOS.map((r) => (
                  <OptionCard
                    key={r.id}
                    active={photo.ratio === r.id}
                    title={r.id}
                    label={r.label}
                    onClick={() => photo.ratio !== r.id && mutatePhoto({ ratio: r.id })}
                  />
                ))}
              </div>
            </div>
          </div>
          <div className="cj-grp cj-actions">
            <button
              className={`cj-btn-outline${dimMode ? " active" : ""}`}
              onClick={() => {
                if (dimMode) setDimMode(false);
                else { setDimMode(true); sliderCtl.current?.toLeft(); }
              }}
            >
              {dimMode ? "Placing points — Esc to exit" : "Add Dimensions"}
            </button>
            <div className="cj-field">
              <button
                className="cj-btn-outline"
                disabled={busy}
                onClick={toggleCover}
                aria-busy={busy}
              >
                {busy ? "Generating Cover…" : coverOn ? "Remove Cover" : "Generate Cover"}
              </button>
              <p className="cj-hint" role="status">
                {busy
                  ? "Hang tight! Your cover will appear below in about 20–40 seconds."
                  : "Adding a cover usually takes 20–40 seconds."}
              </p>
            </div>
          </div>
        </div>
        <div className="cj-foot cj-lst-foot">
          {multi ? (
            <button
              className="cj-btn-primary"
              onClick={() => { setExportSel(new Set(photos.map((_, i) => i))); setExportOpen(true); }}
            >
              Export
            </button>
          ) : (
            <button className="cj-btn-primary" onClick={exportSingle}>Export image</button>
          )}
          {savedNote && <p className="cj-saved">✓ Saved to Library</p>}
        </div>
      </aside>

      {exportOpen && (
        <div className="cj-modal-scrim" onClick={() => setExportOpen(false)}>
          <div className="cj-modal" onClick={(e) => e.stopPropagation()}>
            <div className="cj-modal-head">
              <h2>Export Selection</h2>
            </div>
            <div className="cj-modal-body">
              <div className="cj-modal-grid">
                {photos.map((p, i) => (
                  <button
                    key={entryKey(p)}
                    className={`cj-ex-item${exportSel.has(i) ? " sel" : ""}`}
                    onClick={() =>
                      setExportSel((s) => {
                        const n = new Set(s);
                        if (n.has(i)) n.delete(i); else n.add(i);
                        return n;
                      })
                    }
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={thumbUrl(p, 320)} alt={p.name} />
                    <span className="cj-ex-badge">
                      {exportSel.has(i) ? <CheckCircleIcon size={15} /> : <CircleIcon size={15} />}
                    </span>
                    {/* what this tile actually contains — batch exports pick the right version */}
                    <span className="cj-ex-flags">
                      {p.variant === "cover" && <span className="cj-ex-flag">Cover</span>}
                      {p.dimensions.length > 0 && (
                        <span className="cj-ex-flag">
                          {p.dimensions.length} dim{p.dimensions.length > 1 ? "s" : ""}
                        </span>
                      )}
                    </span>
                  </button>
                ))}
              </div>
            </div>
            <div className="cj-modal-foot">
              <button className="cj-modal-cancel" onClick={() => setExportOpen(false)}>Cancel</button>
              <button className="cj-modal-go" disabled={exportSel.size === 0 || exporting} onClick={exportSelected}>
                {exporting ? "Exporting…" : `Export Selected (${exportSel.size})`}
              </button>
            </div>
          </div>
        </div>
      )}

      {confirmDelete !== null && photos[confirmDelete] && (
        <div className="cj-modal-scrim" onClick={() => setConfirmDelete(null)}>
          <div className="cj-modal small" onClick={(e) => e.stopPropagation()}>
            <div className="cj-modal-head"><h2>Delete photo</h2></div>
            <div className="cj-modal-text">
              Remove <b>{photos[confirmDelete].name}</b> from this batch? Its edits will be lost.
            </div>
            <div className="cj-modal-foot">
              <button className="cj-modal-cancel" onClick={() => setConfirmDelete(null)}>Cancel</button>
              <button className="cj-modal-go" onClick={() => deletePhoto(confirmDelete)}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
