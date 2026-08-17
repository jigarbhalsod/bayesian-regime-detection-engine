import { ArchitectureItem } from "../types/architecture";

interface Props {
  item: ArchitectureItem | null;
  onClose: () => void;
}

function Section({
  title,
  values,
}: {
  title: string;
  values?: string[];
}) {
  if (!values || values.length === 0) return null;

  return (
    <div className="detail-section">
      <h4>{title}</h4>

      {values.map((value, index) => (
        <div className="detail-item" key={index}>
          {value}
        </div>
      ))}
    </div>
  );
}

export default function DetailPanel({ item, onClose }: Props) {
  if (!item) {
    return (
      <aside className="detail-panel empty">
        <div>
          <h2>Project 1A Architecture</h2>
          <p>
            Click any architecture component to inspect its modules,
            entities, procedures, methods, dependencies and execution flow.
          </p>
        </div>
      </aside>
    );
  }

  return (
    <aside className="detail-panel">
      <button className="close-button" onClick={onClose}>
        ×
      </button>

      <div className="detail-layer">{item.layer}</div>

      <h2>{item.name}</h2>

      <div className="detail-meta">
        <span>{item.type}</span>
        {item.status && <span>{item.status}</span>}
        {item.trigger && <span>{item.trigger}</span>}
      </div>

      <p className="detail-description">{item.description}</p>

      {item.frequency && (
        <div className="detail-highlight">
          <strong>Execution:</strong>
          <br />
          {item.frequency}
        </div>
      )}

      <Section title="Inputs" values={item.inputs} />
      <Section title="Outputs" values={item.outputs} />
      <Section title="Modules" values={item.modules} />
      <Section title="Entities" values={item.entities} />
      <Section title="Procedures" values={item.procedures} />
      <Section title="Methods" values={item.methods} />
      <Section title="Dependencies" values={item.dependencies} />
      <Section title="Technology" values={item.technology} />
      <Section title="Security" values={item.security} />
      <Section
        title="Failure Handling"
        values={item.failureHandling}
      />
    </aside>
  );
}