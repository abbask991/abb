import { useState, useMemo } from 'react';
import {
  AlertTriangle,
  Shield,
  Zap,
  Search,
  Clock,
  CheckCircle2,
  XCircle,
  MessageSquare,
  ExternalLink,
} from 'lucide-react';
import { alerts } from '../data/mockData';
import { formatDateTime, relativeTime } from '../components/FormatHelpers';
import type { AlertSeverity, AlertCategory, CaseStatus } from '../types';

export default function Alerts() {
  const [tab, setTab] = useState<'all' | AlertCategory>('all');
  const [severityFilter, setSeverityFilter] = useState<AlertSeverity | ''>('');
  const [statusFilter, setStatusFilter] = useState<CaseStatus | ''>('');
  const [search, setSearch] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const filtered = useMemo(() => {
    return alerts.filter((a) => {
      if (tab !== 'all' && a.category !== tab) return false;
      if (severityFilter && a.severity !== severityFilter) return false;
      if (statusFilter && a.status !== statusFilter) return false;
      if (search && !a.title.toLowerCase().includes(search.toLowerCase()) && !a.description.toLowerCase().includes(search.toLowerCase()))
        return false;
      return true;
    });
  }, [tab, severityFilter, statusFilter, search]);

  const selected = alerts.find((a) => a.id === selectedId) || null;

  const critical = alerts.filter((a) => a.severity === 'critical' && a.status !== 'resolved').length;
  const openCount = alerts.filter((a) => a.status === 'open').length;
  const investigating = alerts.filter((a) => a.status === 'investigating').length;
  const resolved = alerts.filter((a) => a.status === 'resolved').length;

  function severityIcon(severity: AlertSeverity) {
    switch (severity) {
      case 'critical': return <Zap size={16} />;
      case 'high': return <AlertTriangle size={16} />;
      case 'medium': return <Shield size={16} />;
      case 'low': return <Clock size={16} />;
    }
  }

  return (
    <>
      <div className="kpi-grid">
        <div className="kpi-card">
          <span className="kpi-label">Critical</span>
          <span className="kpi-value" style={{ color: 'var(--red)' }}>{critical}</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Open</span>
          <span className="kpi-value" style={{ color: 'var(--yellow)' }}>{openCount}</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Investigating</span>
          <span className="kpi-value" style={{ color: 'var(--blue)' }}>{investigating}</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Resolved</span>
          <span className="kpi-value" style={{ color: 'var(--green)' }}>{resolved}</span>
        </div>
      </div>

      <div className="tabs">
        {(['all', 'financial', 'operational'] as const).map((t) => (
          <div key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'all' ? 'All Alerts' : t === 'financial' ? 'Financial Alerts' : 'Operational Alerts'}
          </div>
        ))}
      </div>

      <div className="filter-bar">
        <div style={{ position: 'relative', flex: 1, maxWidth: 280 }}>
          <Search size={14} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-muted)' }} />
          <input
            type="text"
            placeholder="Search alerts..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ paddingLeft: 30, width: '100%' }}
          />
        </div>
        <select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value as AlertSeverity | '')}>
          <option value="">All Severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as CaseStatus | '')}>
          <option value="">All Statuses</option>
          <option value="open">Open</option>
          <option value="investigating">Investigating</option>
          <option value="resolved">Resolved</option>
          <option value="closed">Closed</option>
        </select>
      </div>

      <div className="grid-2-1">
        <div>
          {filtered.map((alert) => (
            <div
              className="alert-item"
              key={alert.id}
              style={{
                cursor: 'pointer',
                borderColor: selectedId === alert.id ? 'var(--accent)' : undefined,
              }}
              onClick={() => setSelectedId(alert.id)}
            >
              <div className={`alert-icon ${alert.severity}`}>
                {severityIcon(alert.severity)}
              </div>
              <div className="alert-body">
                <div className="alert-title">{alert.title}</div>
                <div className="alert-desc">{alert.description}</div>
                <div className="alert-meta">
                  <span className={`badge badge-${alert.severity}`}>{alert.severity}</span>
                  <span className={`badge badge-${alert.category}`}>{alert.category}</span>
                  <span className={`badge badge-${alert.status}`}>{alert.status}</span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {relativeTime(alert.createdAt)}
                  </span>
                </div>
              </div>
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="empty-state">
              <CheckCircle2 />
              <p>No alerts match your filters</p>
            </div>
          )}
        </div>

        <div>
          {selected ? (
            <div className="card" style={{ position: 'sticky', top: 0 }}>
              <div className="card-header">
                <div className="card-title">Alert Details</div>
                <span className={`badge badge-${selected.severity}`}>{selected.severity}</span>
              </div>

              <div className="detail-row">
                <span className="detail-label">ID</span>
                <span className="detail-value">{selected.id}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Title</span>
                <span className="detail-value">{selected.title}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Category</span>
                <span className="detail-value"><span className={`badge badge-${selected.category}`}>{selected.category}</span></span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Severity</span>
                <span className="detail-value"><span className={`badge badge-${selected.severity}`}>{selected.severity}</span></span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Status</span>
                <span className="detail-value"><span className={`badge badge-${selected.status}`}>{selected.status}</span></span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Created</span>
                <span className="detail-value">{formatDateTime(selected.createdAt)}</span>
              </div>
              {selected.resolvedAt && (
                <div className="detail-row">
                  <span className="detail-label">Resolved</span>
                  <span className="detail-value">{formatDateTime(selected.resolvedAt)}</span>
                </div>
              )}
              <div className="detail-row">
                <span className="detail-label">Linked Txns</span>
                <span className="detail-value">
                  {selected.linkedTransactions.length > 0
                    ? selected.linkedTransactions.join(', ')
                    : 'None'}
                </span>
              </div>

              <div style={{ marginTop: 16 }}>
                <div className="card-subtitle" style={{ marginBottom: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                  Case Management
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button className="btn btn-primary btn-sm">
                    <MessageSquare size={13} /> Open Case
                  </button>
                  <button className="btn btn-ghost btn-sm">
                    <Search size={13} /> Investigate
                  </button>
                  <button className="btn btn-success btn-sm">
                    <CheckCircle2 size={13} /> Resolve
                  </button>
                  <button className="btn btn-danger btn-sm">
                    <XCircle size={13} /> Dismiss
                  </button>
                  <button className="btn btn-ghost btn-sm">
                    <ExternalLink size={13} /> View Txns
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="empty-state">
                <AlertTriangle />
                <p>Select an alert to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
