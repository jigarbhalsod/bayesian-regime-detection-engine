export default function Legend() {
  return (
    <div className="legend">
      <div className="legend-title">Architecture Legend</div>

      <div className="legend-row">
        <span className="legend-dot mvp-dot" />
        MVP
      </div>

      <div className="legend-row">
        <span className="legend-dot future-dot" />
        Future
      </div>

      <div className="legend-row">
        <span className="legend-dot both-dot" />
        MVP + Future
      </div>

      <div className="legend-divider" />

      <div className="legend-row">⚡ Event-driven</div>
      <div className="legend-row">◷ Time-driven</div>
      <div className="legend-row">↗ Request-driven</div>
    </div>
  );
}
