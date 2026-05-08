import { useEffect, useMemo, useState } from "react";
import { Check, Cloud, Cpu, FolderOpen, KeyRound, Save, TriangleAlert } from "lucide-react";

const ROUTES = [
  { id: "planner", label: "Plan model" },
  { id: "builder", label: "Code model" }
];

export function ProviderSettings({ providers, send, snapshot }) {
  const [providerId, setProviderId] = useState("");
  const [model, setModel] = useState("");
  const [groupId, setGroupId] = useState("default");
  const [routeDrafts, setRouteDrafts] = useState({});
  const [workspace, setWorkspace] = useState("");

  useEffect(() => {
    if (!snapshot) return;
    setProviderId(snapshot.provider || "");
    setModel(snapshot.model || "");
    setWorkspace(snapshot.workspace || "");
    const next = {};
    for (const route of snapshot.routes || []) {
      if (route.role === "planner" || route.role === "builder") {
        next[route.role] = { provider: route.provider, model: route.model };
      }
    }
    setRouteDrafts(next);
  }, [snapshot?.provider, snapshot?.model, snapshot?.workspace, snapshot?.routes]);

  const activeProvider = providers.find((provider) => provider.id === providerId);
  const groups = activeProvider?.modelGroups || [];
  const activeGroup = groups.find((group) => group.id === groupId) || groups[0];
  const visibleModels = activeGroup?.models || activeProvider?.models || [];
  const isOllamaCloud = providerId === "ollama" && groupId === "cloud";

  useEffect(() => {
    if (!activeProvider) return;
    const containingGroup = groups.find((group) => group.models.includes(model));
    if (containingGroup) {
      setGroupId(containingGroup.id);
    } else {
      setGroupId(groups[0]?.id || "default");
    }
  }, [activeProvider?.id, model]);

  const authState = snapshot?.authOk ? "Ready" : "Needs auth";

  return (
    <main className="settings-shell">
      <header className="settings-header">
        <div>
          <h1>Providers</h1>
          <p>Choose the active engine, discover relevant models, and set simple plan/code routing.</p>
        </div>
        <StatusBadge ok={snapshot?.authOk} label={authState} />
      </header>

      <section className="provider-grid">
        <div className="provider-list-panel">
          {providers.map((provider) => (
            <button
              key={provider.id}
              className={provider.id === providerId ? "provider-tile active" : "provider-tile"}
              type="button"
              onClick={() => {
                setProviderId(provider.id);
                const firstGroup = provider.modelGroups?.[0];
                setGroupId(firstGroup?.id || "default");
                setModel(firstGroup?.models?.[0] || provider.models?.[0] || "");
              }}
            >
              <Cloud size={16} />
              <span>
                <strong>{provider.label}</strong>
                <small>{provider.status === "construction" ? "In construction" : "Ready"}</small>
              </span>
            </button>
          ))}
        </div>

        <div className="provider-detail-panel">
          <div className="detail-title">
            <Cpu size={18} />
            <div>
              <h2>{activeProvider?.label || "Select provider"}</h2>
              <p>{isOllamaCloud ? "Cloud models are available through Ollama auth or compatible local Ollama routing." : activeProvider?.note || "Only relevant models are shown for this provider."}</p>
            </div>
          </div>

          {groups.length > 1 ? (
            <div className="group-tabs">
              {groups.map((group) => (
                <button key={group.id} className={group.id === groupId ? "active" : ""} type="button" onClick={() => {
                  setGroupId(group.id);
                  setModel(group.models[0] || "");
                }}>
                  {group.label}
                </button>
              ))}
            </div>
          ) : null}

          <div className="model-grid">
            {visibleModels.map((item) => (
              <button key={item} type="button" className={item === model ? "model-tile active" : "model-tile"} onClick={() => setModel(item)}>
                <span>{item}</span>
                {item === model ? <Check size={15} /> : null}
              </button>
            ))}
          </div>

          <button className="primary-button wide" type="button" disabled={!providerId || !model} onClick={() => send({ type: "setProviderModel", provider: providerId, model })}>
            <Save size={16} />
            Apply active engine
          </button>
        </div>
      </section>

      <section className="routing-card">
        <h2>Task routing</h2>
        <div className="route-settings">
          {ROUTES.map((route) => (
            <RouteEditor
              key={route.id}
              providers={providers}
              route={route}
              value={routeDrafts[route.id] || { provider: providerId, model }}
              onChange={(value) => setRouteDrafts((drafts) => ({ ...drafts, [route.id]: value }))}
              onSave={() => {
                const value = routeDrafts[route.id] || { provider: providerId, model };
                send({ type: "setRoute", role: route.id, provider: value.provider, model: value.model });
              }}
            />
          ))}
        </div>
      </section>

      <section className="workspace-card">
        <h2>Workspace</h2>
        <div className="workspace-row">
          <input value={workspace} onChange={(event) => setWorkspace(event.target.value)} />
          <button className="secondary-button" type="button" onClick={async () => {
            const path = await window.crypt?.chooseDirectory?.();
            if (path) {
              setWorkspace(path);
              send({ type: "setWorkspace", path });
            }
          }}>
            <FolderOpen size={15} />
            Browse
          </button>
          <button className="secondary-button" type="button" onClick={() => send({ type: "setWorkspace", path: workspace })}>Save</button>
        </div>
      </section>
    </main>
  );
}

function RouteEditor({ providers, route, value, onChange, onSave }) {
  const provider = providers.find((item) => item.id === value.provider) || providers[0];
  const models = useMemo(() => provider?.modelGroups?.flatMap((group) => group.models) || provider?.models || [], [provider]);

  return (
    <div className="route-editor">
      <strong>{route.label}</strong>
      <select value={value.provider} onChange={(event) => {
        const nextProvider = providers.find((item) => item.id === event.target.value);
        const nextModel = nextProvider?.models?.[0] || nextProvider?.modelGroups?.[0]?.models?.[0] || "";
        onChange({ provider: event.target.value, model: nextModel });
      }}>
        {providers.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
      </select>
      <select value={value.model} onChange={(event) => onChange({ ...value, model: event.target.value })}>
        {models.map((item) => <option key={item} value={item}>{item}</option>)}
      </select>
      <button className="secondary-button" type="button" onClick={onSave}>Save</button>
    </div>
  );
}

function StatusBadge({ ok, label }) {
  return (
    <div className={ok ? "status-badge ok" : "status-badge warn"}>
      {ok ? <KeyRound size={15} /> : <TriangleAlert size={15} />}
      {label}
    </div>
  );
}
