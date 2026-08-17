interface Props {
  search: string;
  setSearch: (value: string) => void;
}

export default function Toolbar({ search, setSearch }: Props) {
  return (
    <div className="toolbar">
      <div>
        <div className="toolbar-title">
          Zetheta 1A
        </div>

        <div className="toolbar-subtitle">
          Market Regime Detection & Equity Direction Forecasting
        </div>
      </div>

      <input
        className="search"
        placeholder="Search architecture..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
    </div>
  );
}