"use client";

/* New Listing's state + processing pipeline, lifted out of the page into a
   layout-level provider. Layouts survive route changes (the page component
   does not), so uploads keep processing and results stay put while the user
   visits Social or the Library and comes back.

   The page (app/page.tsx) is now purely presentational: it reads everything
   from useListing() and keeps only DOM refs and hover/drag visual state. */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { Dimension } from "@/components/DimensionOverlay";
import { saveBlob } from "./save-file";

export type Ratio = "1:1" | "4:3" | "4:5";
export const RATIOS: { id: Ratio; label: string }[] = [
  { id: "1:1", label: "EBAY" },
  { id: "4:3", label: "WEBSITE" },
  { id: "4:5", label: "INSTAGRAM" },
];
export const RATIO_AR: Record<Ratio, number> = { "1:1": 1, "4:3": 4 / 3, "4:5": 4 / 5 };

export type Photo = {
  id: string;
  name: string;
  ratio: Ratio;
  dimensions: Dimension[];
  variant?: "cover";           // undefined = white-plate entry
  gen?: number;                // cover generation stamp — cache-buster, since
                               // remove+regenerate reuses the same (id, kind)
};

export type ProcRow = {
  name: string;
  thumb: string | null;
  state: "waiting" | "active" | "done" | "error";
  warn?: string;
};

/* A failed upload kept in the filmstrip so it can be retried in place —
   the File handle is retained for the retry request. */
export type FailedRow = {
  name: string;
  thumb: string | null;
  file: File;
  warn?: string;
  busy?: boolean;
};

/** Stable per-entry key — cover entries share the backend id with their
 *  white-plate sibling, so the id alone would collide. */
export const entryKey = (p: Photo) => `${p.id}:${p.variant ?? "after"}`;

export const imgUrl = (p: Photo, slot: "after" | "before", w?: number) => {
  const kind = slot === "after" && p.variant === "cover" ? "cover" : slot;
  return `/api/testing2/${p.id}/image?kind=${kind}&ratio=${encodeURIComponent(p.ratio)}` +
    (w ? `&w=${w}` : "") + (p.gen ? `&v=${p.gen}` : "");
};

export const thumbUrl = (p: Photo, s?: number) =>
  `/api/testing2/${p.id}/thumb?kind=${p.variant === "cover" ? "cover" : "after"}` +
  (s ? `&s=${s}` : "") + (p.gen ? `&v=${p.gen}` : "");

/* Batch uploads run through gpt-image-2 (~15-50s each), so process several in
   parallel. 3 workers balances total wall time against OpenAI's per-minute
   rate limits (the backend retries 429s with backoff as a safety net). */
const CONCURRENCY = 3;

async function runPool<T>(jobs: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const results = new Array<T>(jobs.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(limit, jobs.length) }, async () => {
      while (next < jobs.length) {
        const i = next++;
        results[i] = await jobs[i]();
      }
    }),
  );
  return results;
}

async function processFile(f: File): Promise<{ id?: string; warn?: string }> {
  for (let attempt = 0; attempt < 2; attempt++) {
    const fd = new FormData();
    fd.append("file", f);
    try {
      const res = await fetch("/api/testing2", { method: "POST", body: fd });
      const j = await res.json();
      if (!res.ok || !j.id) return { warn: j.warn ?? `${f.name}: processing failed` };
      return { id: j.id };
    } catch {
      if (attempt === 1) return { warn: `${f.name}: network error` };
    }
  }
  return { warn: `${f.name}: network error` };
}

/* Shown whenever a photo's server-side record is gone (the in-memory backend
   was restarted) — the only fix is re-uploading. */
export const SESSION_EXPIRED =
  "Session expired — the server restarted and these photos are gone. Please re-upload.";

async function downloadExport(items: object[], fallback: string) {
  const res = await fetch("/api/testing2/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  if (!res.ok) throw new Error(res.status === 404 ? SESSION_EXPIRED : `Export failed (${res.status}) — please retry.`);
  const blob = await res.blob();
  const m = (res.headers.get("Content-Disposition") ?? "").match(/filename="([^"]+)"/);
  await saveBlob(blob, m?.[1] ?? fallback);
}

/* The size the result stage requests — prefetches must match it exactly for
   the browser cache to be hit. Phones get a lighter render: 1400px is wasted
   on a ~1100-device-pixel screen and mobile bandwidth is the bottleneck.
   Only called client-side (the result stage never server-renders). */
export const stageW = () =>
  typeof window !== "undefined" && window.matchMedia?.("(pointer: coarse)").matches
    ? 1000
    : 1400;

function useListingState() {
  const [phase, setPhase] = useState<"home" | "processing" | "result">("home");
  const [rows, setRows] = useState<ProcRow[]>([]);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [active, setActive] = useState(0);
  const [warn, setWarn] = useState<string | null>(null);
  const [animate, setAnimate] = useState(false);
  const [imgLoading, setImgLoading] = useState(false);
  const [pending, setPending] = useState<ProcRow[]>([]);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportSel, setExportSel] = useState<Set<number>>(new Set());
  const [exporting, setExporting] = useState(false);
  const [dimMode, setDimMode] = useState(false);
  const [coverBusy, setCoverBusy] = useState<Set<string>>(new Set());
  const [failed, setFailed] = useState<FailedRow[]>([]);
  const [savedNote, setSavedNote] = useState(false);
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Covers that finished while the user was on another photo — badge, don't yank.
  const [newCovers, setNewCovers] = useState<Set<string>>(new Set());
  // Soft delete: the removed entry is held here for 5s before the backend
  // delete really happens, so a slip of the mouse can't destroy paid work.
  const [undoDelete, setUndoDelete] = useState<{ entry: Photo; index: number } | null>(null);
  const undoTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  /* Transient "Saved to Library" confirmation after a successful export. */
  const flashSaved = useCallback(() => {
    setSavedNote(true);
    if (savedTimer.current) clearTimeout(savedTimer.current);
    savedTimer.current = setTimeout(() => setSavedNote(false), 2500);
  }, []);
  const sliderCtl = useRef<{ toLeft: () => void } | null>(null);
  const cancelled = useRef(false);

  const photo = photos[active];

  // Live mirror for async callbacks (delayed deletes).
  const photosRef = useRef(photos);
  useEffect(() => {
    photosRef.current = photos;
  });

  /* Fire-and-forget image warmup. The browser caches the response, so when
     the visible <img> asks for the same URL it paints instantly instead of
     waiting a full compose + download round trip. */
  const prefetched = useRef<Set<string>>(new Set());
  const prefetch = useCallback((url: string) => {
    if (prefetched.current.has(url)) return;
    prefetched.current.add(url);
    const img = new Image();
    img.src = url;
  }, []);

  /* Warm a fresh upload's stage + filmstrip images the moment its id lands,
     while the rest of the batch is still processing — by the time the result
     screen appears these are already in the browser cache. */
  const prefetchPhoto = useCallback((id: string) => {
    const p: Photo = { id, name: "", ratio: "1:1", dimensions: [] };
    prefetch(imgUrl(p, "after", stageW()));
    prefetch(imgUrl(p, "before", stageW()));
    prefetch(thumbUrl(p));
  }, [prefetch]);

  /* Warm the other ratios for the active photo in the background, so
     switching Output Ratio is a browser-cache hit instead of a fresh
     server-side compose + full-size download. Waits until the visible stage
     image has finished (imgLoading false) — kicking these off earlier makes
     them compete with it for bandwidth and slows the first paint. */
  useEffect(() => {
    if (phase !== "result" || !photo || imgLoading) return;
    for (const r of RATIOS) {
      if (r.id === photo.ratio) continue;
      prefetch(imgUrl({ ...photo, ratio: r.id }, "after", stageW()));
      prefetch(imgUrl({ ...photo, ratio: r.id }, "before", stageW()));
    }
  }, [phase, photo, imgLoading, prefetch]);

  // A fresh cover was selected — its "new" badge has served its purpose.
  useEffect(() => {
    if (photo?.variant === "cover" && newCovers.has(photo.id)) {
      setNewCovers((s) => { const n = new Set(s); n.delete(photo.id); return n; });
    }
  }, [photo, newCovers]);

  /* Return to the empty upload screen. Fired by the sidebar's New Listing item —
     clicking it while already on "/" can't remount this page, so it dispatches
     a "cj:reset-listing" event that we handle here. */
  const resetToHome = useCallback(() => {
    cancelled.current = true;
    setPhase("home");
    setRows([]);
    setPhotos([]);
    setPending([]);
    setActive(0);
    setWarn(null);
    setDimMode(false);
    setExportOpen(false);
    setConfirmDelete(null);
    setCoverBusy(new Set());
    setFailed([]);
    setNewCovers(new Set());
    if (undoTimer.current) clearTimeout(undoTimer.current);
    setUndoDelete(null);
  }, []);

  // Esc leaves dimension-placing mode (matches the Library's Esc habit).
  useEffect(() => {
    if (!dimMode) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setDimMode(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dimMode]);

  useEffect(() => {
    window.addEventListener("cj:reset-listing", resetToHome);
    return () => window.removeEventListener("cj:reset-listing", resetToHome);
  }, [resetToHome]);

  const mutatePhoto = useCallback((patch: Partial<Photo>) => {
    setImgLoading(true);
    setPhotos((ps) => ps.map((p, i) => (i === active ? { ...p, ...patch } : p)));
  }, [active]);

  const startBatch = useCallback(async (files: File[]) => {
    files = files.filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    cancelled.current = false;
    setRows(files.map((f) => ({ name: f.name, thumb: URL.createObjectURL(f), state: "waiting" })));
    setPhase("processing");

    const results = await runPool(
      files.map((f, i) => async () => {
        if (cancelled.current) return null;
        setRows((r) => r.map((row, j) => (j === i ? { ...row, state: "active" } : row)));
        const out = await processFile(f);
        if (out.id) prefetchPhoto(out.id);
        setRows((r) => r.map((row, j) =>
          (j === i ? { ...row, state: out.id ? "done" : "error", warn: out.warn } : row)));
        return out;
      }),
      CONCURRENCY,
    );
    if (cancelled.current) return;

    const got: Photo[] = [];
    const fails: FailedRow[] = [];
    const warns: string[] = [];
    results.forEach((out, i) => {
      if (out?.id) {
        got.push({ id: out.id, name: files[i].name, ratio: "1:1", dimensions: [] });
      } else if (out) {
        fails.push({ name: files[i].name, thumb: URL.createObjectURL(files[i]), file: files[i], warn: out.warn });
        warns.push(`${files[i].name}: ${out.warn ?? "failed"}`);
      }
    });
    setFailed(fails);
    setWarn(warns.length ? warns.join("  ") : null);
    if (got.length) {
      setPhotos(got);
      setActive(0);
      setAnimate(true);
      // The stage image may still be in flight (prefetch warms it, but can't
      // guarantee it) — show the loading veil until CompareSlider reports in,
      // instead of a blank stage.
      setImgLoading(true);
      setPhase("result");
    } else {
      setPhase("home");
    }
  }, [prefetchPhoto]);

  const cancelBatch = useCallback(() => {
    cancelled.current = true;
    setPhase("home");
  }, []);

  const addMore = useCallback(async (files: File[]) => {
    files = files.filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    setPending(files.map((f) => ({ name: f.name, thumb: URL.createObjectURL(f), state: "waiting" })));
    const results = await runPool(
      files.map((f, i) => async () => {
        setPending((t) => t.map((x, j) => (j === i ? { ...x, state: "active" } : x)));
        const out = await processFile(f);
        if (out.id) prefetchPhoto(out.id);
        setPending((t) => t.map((x, j) => (j === i ? { ...x, state: "done" } : x)));
        return out;
      }),
      CONCURRENCY,
    );
    const warns: string[] = [];
    results.forEach((out, i) => {
      if (out.id) {
        setPhotos((ps) => [...ps, { id: out.id!, name: files[i].name, ratio: "1:1", dimensions: [] }]);
      } else {
        setFailed((fs) => [...fs, {
          name: files[i].name, thumb: URL.createObjectURL(files[i]), file: files[i], warn: out.warn,
        }]);
        warns.push(`${files[i].name}: ${out.warn ?? "failed"}`);
      }
    });
    setPending([]);
    setWarn(warns.length ? warns.join("  ") : null);
  }, [prefetchPhoto]);

  /* Retry a failed upload in place — the tile spins, then either becomes a
     real photo or stays with an updated message. */
  const retryFailed = useCallback(async (idx: number) => {
    const row = failed[idx];
    if (!row || row.busy) return;
    setFailed((fs) => fs.map((f, j) => (j === idx ? { ...f, busy: true } : f)));
    const out = await processFile(row.file);
    if (out.id) {
      prefetchPhoto(out.id);
      setPhotos((ps) => [...ps, { id: out.id!, name: row.name, ratio: "1:1", dimensions: [] }]);
      setFailed((fs) => fs.filter((f) => f.file !== row.file));
      setWarn(null);
    } else {
      setFailed((fs) => fs.map((f, j) => (j === idx ? { ...f, busy: false, warn: out.warn } : f)));
      setWarn(`${row.name}: ${out.warn ?? "failed"}`);
    }
  }, [failed, prefetchPhoto]);

  /* Cover on/off for the active photo's backend id. On: one-time generation
     (classify + scene, cached server-side) then a new filmstrip entry; off:
     remove the entry (cache stays, re-enabling is free). */
  const toggleCover = useCallback(async () => {
    if (!photo) return;
    const id = photo.id;
    const existing = photos.findIndex((p) => p.id === id && p.variant === "cover");
    if (existing >= 0) {
      // Destructive by design ("Removing a cover cannot be undone"): drop the
      // server-side cache too, or the next Generate would silently return the
      // same image from the POST route's cache instead of a fresh run.
      fetch(`/api/testing2/${id}/cover`, { method: "DELETE" }).catch(() => {});
      setPhotos((ps) => ps.filter((_, j) => j !== existing));
      setActive((a) => {
        const base = photos.findIndex((p) => p.id === id && !p.variant);
        return a === existing ? Math.max(0, base) : a > existing ? a - 1 : a;
      });
      return;
    }
    if (coverBusy.has(id)) return;
    setCoverBusy((s) => new Set(s).add(id));
    try {
      const res = await fetch(`/api/testing2/${id}/cover`, { method: "POST" });
      const j = await res.json().catch(() => ({}));
      if (!res.ok || !j.ok) {
        setWarn(res.status === 404 ? SESSION_EXPIRED : (j.warn ?? "Cover generation failed — please retry."));
        return;
      }
      setPhotos((ps) => {
        const base = ps.findIndex((p) => p.id === id && !p.variant);
        if (base < 0 || ps.some((p) => p.id === id && p.variant === "cover")) return ps;
        const entry: Photo = {
          id, name: `${ps[base].name} · cover`, ratio: ps[base].ratio,
          dimensions: [], variant: "cover", gen: Date.now(),
        };
        return [...ps.slice(0, base + 1), entry, ...ps.slice(base + 1)];
      });
      // Never yank focus: the new cover lands in the filmstrip with an
      // "unread" dot, cleared when the user clicks it.
      setNewCovers((s) => new Set(s).add(id));
    } catch {
      setWarn("Cover generation failed: network error");
    } finally {
      setCoverBusy((s) => { const n = new Set(s); n.delete(id); return n; });
    }
  }, [photo, photos, coverBusy]);

  /* Backend delete only when no remaining filmstrip entry shares the record
     (cover + white-plate entries share one id). Checked against live state so
     an undo that restored the entry cancels the wipe. */
  const finalizeDelete = useCallback((entry: Photo) => {
    if (!photosRef.current.some((p) => p.id === entry.id)) {
      fetch(`/api/testing2/${entry.id}`, { method: "DELETE" }).catch(() => {});
    }
  }, []);

  const deletePhoto = useCallback((i: number) => {
    const target = photos[i];
    if (!target) return;
    const next = photos.filter((_, j) => j !== i);
    // A new delete finalizes any previous pending one.
    if (undoTimer.current) { clearTimeout(undoTimer.current); undoTimer.current = null; }
    if (undoDelete) finalizeDelete(undoDelete.entry);
    setPhotos(next);
    setConfirmDelete(null);
    if (!next.length) {
      // Removing the last photo leaves the page — no toast to host an undo,
      // so this one deletes immediately (the confirm dialog still gates it).
      setUndoDelete(null);
      fetch(`/api/testing2/${target.id}`, { method: "DELETE" }).catch(() => {});
      setPhase("home");
      setActive(0);
      return;
    }
    setActive((a) => Math.max(0, Math.min(i < a ? a - 1 : a, next.length - 1)));
    setUndoDelete({ entry: target, index: i });
    undoTimer.current = setTimeout(() => {
      finalizeDelete(target);
      setUndoDelete(null);
      undoTimer.current = null;
    }, 5000);
  }, [photos, undoDelete, finalizeDelete]);

  const undoRemove = useCallback(() => {
    if (undoTimer.current) { clearTimeout(undoTimer.current); undoTimer.current = null; }
    const pending = undoDelete;
    setUndoDelete(null);
    if (!pending) return;
    // setPhotos must not run inside the setUndoDelete updater: updaters must
    // be pure, and StrictMode double-invokes them — the nested call inserted
    // the entry twice. The duplicate guard also covers double-clicked Undo.
    setPhotos((ps) => {
      if (ps.some((p) => entryKey(p) === entryKey(pending.entry))) return ps;
      const idx = Math.min(pending.index, ps.length);
      return [...ps.slice(0, idx), pending.entry, ...ps.slice(idx)];
    });
  }, [undoDelete]);

  const exportSelected = useCallback(async () => {
    const items = photos
      .filter((_, i) => exportSel.has(i))
      .map((p) => ({ id: p.id, ratio: p.ratio, kind: p.variant === "cover" ? "cover" : "after", dimensions: p.dimensions }));
    if (!items.length) return;
    setExporting(true);
    try {
      await downloadExport(items, items.length > 1 ? "cj_photos.zip" : "cj_photo.png");
      setExportOpen(false);
      flashSaved();
    } catch (e) {
      setWarn(e instanceof Error ? e.message : "Export failed — please retry.");
    } finally {
      setExporting(false);
    }
  }, [photos, exportSel, flashSaved]);

  const exportSingle = useCallback(() => {
    if (!photo) return;
    downloadExport(
      [{ id: photo.id, ratio: photo.ratio, kind: photo.variant === "cover" ? "cover" : "after", dimensions: photo.dimensions }],
      `cj_listing_${photo.ratio.replace(":", "x")}.png`,
    ).then(flashSaved)
      .catch((e) => setWarn(e instanceof Error ? e.message : "Export failed — please retry."));
  }, [photo, flashSaved]);

  const donePct = useMemo(() => {
    const done = rows.filter((r) => r.state === "done" || r.state === "error").length;
    return rows.length ? Math.round((done / rows.length) * 100) : 0;
  }, [rows]);

  return {
    phase, rows, photos, active, warn, animate, imgLoading, pending,
    confirmDelete, exportOpen, exportSel, exporting, dimMode, coverBusy,
    failed, savedNote, newCovers, undoDelete, photo, donePct, sliderCtl,
    setPhotos, setActive, setWarn, setAnimate, setImgLoading, setDimMode,
    setConfirmDelete, setExportOpen, setExportSel,
    startBatch, cancelBatch, addMore, retryFailed, toggleCover,
    deletePhoto, undoRemove, exportSelected, exportSingle, mutatePhoto,
  };
}

type ListingState = ReturnType<typeof useListingState>;

const ListingContext = createContext<ListingState | null>(null);

export function ListingProvider({ children }: { children: ReactNode }) {
  const value = useListingState();
  return <ListingContext.Provider value={value}>{children}</ListingContext.Provider>;
}

export function useListing(): ListingState {
  const ctx = useContext(ListingContext);
  if (!ctx) throw new Error("useListing must be used inside <ListingProvider>");
  return ctx;
}
