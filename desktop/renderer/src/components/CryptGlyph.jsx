const GLYPHS = {
  sigil: (
    <>
      <path className="cg-aura" d="M12 3.8 20.2 8.4v9.2L12 22.2 3.8 17.6V8.4L12 3.8Z" />
      <path d="M12.4 2.8 7.2 11h4l-2 10.2 7.6-11h-4.2l2.8-7.4Z" />
      <path d="M5.8 8.8 2.6 7.2M18.2 8.8l3.2-1.6M5.8 17.2l-3.2 1.6M18.2 17.2l3.2 1.6" />
      <circle cx="12" cy="12" r="1.35" />
    </>
  ),
  forge: (
    <>
      <path d="M6.2 18.8 17.8 7.2" />
      <path d="m14.9 4.3 4.8 4.8" />
      <path d="M4.5 20.5h5.8l1.8-3.6-3-3-3.7 1.8Z" />
      <path d="M13 14.2h5.7M15.9 11.4v5.7" />
    </>
  ),
  hunt: (
    <>
      <path d="M8.2 6.9 5.8 4.5M15.8 6.9l2.4-2.4" />
      <path d="M6.7 11.3h10.6M5.4 15.1h13.2" />
      <path d="M8.2 8.1h7.6l1.7 4.2-1 6-4.5 2.4-4.5-2.4-1-6Z" />
      <path d="M10.1 12.3h.1M13.8 12.3h.1M12 15v3.4" />
    </>
  ),
  beauty: (
    <>
      <path d="M12 3.8c4.1 1.5 6.2 4.1 6.2 7.8 0 4.8-3.7 8.6-8.8 8.6-2.1 0-3.6-.7-3.6-2.1 0-1.3 1-2.1 2.2-2.1h1.7c.9 0 1.2-.8.7-1.5-.5-.8-.8-1.8-.8-3 0-3.5 2.1-6 6.4-7.7Z" />
      <circle cx="9.2" cy="10.4" r=".55" />
      <circle cx="12.1" cy="8.5" r=".55" />
      <circle cx="14.8" cy="10.7" r=".55" />
      <path d="M14 15.9h4.2M16.1 13.8V18" />
    </>
  ),
  boss: (
    <>
      <path d="M12 3.7 19 7v5.4c0 4.2-2.8 7.2-7 8.9-4.2-1.7-7-4.7-7-8.9V7Z" />
      <path d="m8.4 12 2.4 2.4 5-5" />
      <path d="M7.3 7.8 12 5.6l4.7 2.2" />
    </>
  ),
  ship: (
    <>
      <path d="M13.9 4.2 20 10.3l-5.1 1.5-2.7 5.7-2.1-3.6-3.6-2.1 5.7-2.7Z" />
      <path d="m6.6 16.2-2.3 3.5 3.5-2.3M10.5 14.5l-2.3 3.4" />
      <circle cx="14.7" cy="9.3" r="1.1" />
    </>
  ),
  lore: (
    <>
      <path d="M7.2 4.7h9.6c1.1 0 2 .9 2 2v12.6H8.1c-1.6 0-2.9-1.1-2.9-2.6V6.7c0-1.1.9-2 2-2Z" />
      <path d="M8.1 19.3c-1.6 0-2.9-1.1-2.9-2.6s1.3-2.6 2.9-2.6h10.7" />
      <path d="M8.5 8.4h6.9M8.5 11.2H14" />
      <path d="M15.4 15.8h1.7" />
    </>
  ),
  shard: (
    <>
      <path d="M12 2.9 18.2 12 12 21.1 5.8 12Z" />
      <path d="M12 2.9V12l6.2 0M12 12v9.1L5.8 12Z" />
    </>
  )
};

export function CryptGlyph({ className = "", name = "sigil", size = 24, title }) {
  const glyph = GLYPHS[name] || GLYPHS.sigil;
  return (
    <svg
      aria-hidden={title ? undefined : true}
      aria-label={title}
      className={`crypt-glyph ${className}`.trim()}
      fill="none"
      height={size}
      role={title ? "img" : undefined}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.75"
      viewBox="0 0 24 24"
      width={size}
    >
      {glyph}
    </svg>
  );
}
