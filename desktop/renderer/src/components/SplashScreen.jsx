import { CryptGlyph } from "./CryptGlyph.jsx";

const BOOT_LINES = [
  "local engine",
  "workspace memory",
  "tool bus",
  "preview stage"
];

export function SplashScreen({ leaving = false, onSkip }) {
  return (
    <div className={leaving ? "splash-screen leaving" : "splash-screen"} onClick={onSkip} role="presentation">
      <div className="splash-flash" aria-hidden="true" />
      <svg className="splash-plasma" viewBox="0 0 1440 920" aria-hidden="true">
        <defs>
          <filter id="crypt-electric-glow" x="-35%" y="-35%" width="170%" height="170%">
            <feGaussianBlur stdDeviation="6" result="blur" />
            <feColorMatrix
              in="blur"
              result="blueGlow"
              type="matrix"
              values="0 0 0 0 0.18 0 0 0 0 0.74 0 0 0 0 1 0 0 0 0.92 0"
            />
            <feMerge>
              <feMergeNode in="blueGlow" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <g className="plasma-main" filter="url(#crypt-electric-glow)">
          <path className="plasma-path plasma-path-1" pathLength="1" d="M720 458 C690 414 724 386 690 352 C650 313 684 276 626 226" />
          <path className="plasma-path plasma-path-2" pathLength="1" d="M724 460 C782 420 776 378 844 348 C918 315 928 252 1008 214" />
          <path className="plasma-path plasma-path-3" pathLength="1" d="M718 462 C656 488 636 542 570 560 C500 579 460 640 392 662" />
          <path className="plasma-path plasma-path-4" pathLength="1" d="M725 462 C800 488 830 540 904 552 C994 566 1040 636 1130 670" />
          <path className="plasma-path plasma-path-5" pathLength="1" d="M718 456 C720 522 750 560 730 628 C708 704 746 760 728 836" />
        </g>
        <g className="plasma-threads" filter="url(#crypt-electric-glow)">
          <path pathLength="1" d="M720 460 C704 432 690 416 668 402 C644 386 622 382 596 364" />
          <path pathLength="1" d="M720 460 C752 446 772 430 790 402 C806 374 838 362 866 338" />
          <path pathLength="1" d="M720 460 C710 494 690 510 658 526 C626 542 612 568 588 596" />
          <path pathLength="1" d="M720 460 C756 478 776 500 802 530 C834 566 872 574 910 608" />
          <path pathLength="1" d="M720 460 C735 434 736 414 728 384 C720 350 740 326 750 298" />
          <path pathLength="1" d="M720 460 C696 476 682 492 676 520 C668 556 638 572 612 596" />
        </g>
      </svg>
      <div className="splash-noise" aria-hidden="true" />
      <div className="splash-rings" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <div className="splash-shards" aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
        <span />
        <span />
      </div>
      <div className="splash-bolts" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
        <i />
      </div>
      <main className="splash-core" aria-label="Crypt loading">
        <div className="splash-sigil">
          <CryptGlyph name="sigil" size={72} />
        </div>
        <div className="splash-copy">
          <span>local-first workbench</span>
          <h1>Crypt</h1>
          <p>Private engine coming online.</p>
        </div>
        <div className="splash-boot">
          {BOOT_LINES.map((line) => (
            <div key={line}>
              <span>{line}</span>
              <strong>armed</strong>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
