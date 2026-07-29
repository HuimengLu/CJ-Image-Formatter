"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ApertureIcon, ArchiveIcon, ImagesSquareIcon, TagIcon } from "@/components/icons";

const items = [
  { href: "/", icon: <TagIcon />, label: "New Listing" },
  { href: "/social", icon: <ApertureIcon />, label: "Social" },
  { href: "/library", icon: <ImagesSquareIcon />, label: "Library" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const resetIfCurrent = (href: string) => {
    // Only a re-click on the CURRENT page means "start over". Listing state
    // now lives in a layout-level provider that survives navigation, so an
    // unconditional reset here would wipe results on every return visit.
    if (href === "/" && pathname === "/") {
      window.dispatchEvent(new Event("cj:reset-listing"));
    } else if (href === "/legacy" && pathname === "/legacy") {
      window.dispatchEvent(new Event("cj:reset-legacy"));
    }
  };
  return (
    <>
      <div id="cj-side">
        <div className="cj-brand">CJ&nbsp;Studio</div>
        <div className="cj-navcol">
          {items.map((it) => (
            <Link
              key={it.label}
              href={it.href}
              className={`cj-item${pathname === it.href ? " active" : ""}`}
              // The New Listing item returns to its upload screen. When already
              // on "/", Link can't remount the page, so signal it to reset.
              onClick={() => resetIfCurrent(it.href)}
            >
              {it.icon}
              <span>{it.label}</span>
            </Link>
          ))}
        </div>
        {/* Archived tools, pinned at the bottom and muted. */}
        <div>
          <div className="cj-tipwrap">
            <Link
              href="/legacy"
              className={`cj-item cj-item-muted${pathname === "/legacy" ? " active" : ""}`}
              onClick={() => resetIfCurrent("/legacy")}
            >
              <ArchiveIcon />
              <span>Legacy Listing</span>
            </Link>
            <div className="cj-tooltip" role="tooltip">
              <span className="cj-tooltip-title">Legacy Listing</span>
              <span className="cj-tooltip-desc">Runs on this machine — no AI credits used</span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile top app bar (≤900px; the sidebar hides there). The active tab
          identifies the page, so the per-page title bar is hidden on mobile. */}
      <nav id="cj-topbar" aria-label="Primary">
        <span className="cj-top-brand">CJ&nbsp;Studio</span>
        {items.map((it) => (
          <Link
            key={it.label}
            href={it.href}
            className={`cj-top-item${pathname === it.href ? " active" : ""}`}
            aria-current={pathname === it.href ? "page" : undefined}
            // Keeps the accessible name when narrow screens hide the label.
            aria-label={it.label === "New Listing" ? "Listing" : it.label}
            onClick={() => resetIfCurrent(it.href)}
          >
            {it.icon}
            <span aria-hidden="true">{it.label === "New Listing" ? "Listing" : it.label}</span>
          </Link>
        ))}
        {/* No Legacy entry here: it's an archived tool staff use at the desk,
            and the mobile bar has no room for a muted archival link. /legacy
            still works by URL on mobile. */}
      </nav>
    </>
  );
}
