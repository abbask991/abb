import { useState, useMemo } from 'react';
import { CheckCircle2, AlertCircle, HelpCircle, Search, FileText, ThumbsUp, ThumbsDown } from 'lucide-react';
import { reconciliationCases } from '../data/mockData';
import { formatCurrency, formatDateTime } from '../components/FormatHelpers';
import type { ReconciliationStatus, CaseStatus } from '../types';

export default function Reconciliation() {
  const [tab, setTab] = useState<'all' | ReconciliationStatus>('all');
  const [caseFilter, setCaseFilter] = useState<CaseStatus | ''>('');
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    return reconciliationCases.filter((r) => {
      if (tab !== 'all' && r.status !== tab) return false;
      if (caseFilter && r.caseStatus !== caseFilter) return false;
      if (search && !r.id.toLowerCase().includes(search.toLowerCase()) && !r.transactionId.toLowerCase().includes(search.toLowerCase()))
        return false;
      return true;
    });
  }, [tab, caseFilter, search]);

  const matched = reconciliationCases.filter((r) => r.status === 'matched').length;
  const partial = reconciliationCases.filter((r) => r.status === 'partial').length;
  const exceptions = reconciliationCases.filter((r) => r.status === 'exception').length;

  const summaryCards = [
    { label: 'Matched', count: matched, icon: CheckCircle2, color: 'var(--green)', bg: 'var(--green-bg)' },
    { label: 'Partial', count: partial, icon: HelpCircle, color: 'var(--yellow)', bg: 'var(--yellow-bg)' },
    { label: 'Exceptions', count: exceptions, icon: AlertCircle, color: 'var(--red)', bg: 'var(--red-bg)' },
  ];

  function getMatchColor(score: number) {
    if (score >= 90) return 'var(--green)';
    if (score >= 60) return 'var(--yellow)';
    return 'var(--red)';
  }

  return (
    <>
      <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        {summaryCards.map((s) => (
          <div className="kpi-card" key={s.label}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span className="kpi-label">{s.label}</span>
              <div style={{ width: 34, height: 34, borderRadius: 8, background: s.bg, display: 'flex', alignItems: 'center', justifyContent: 'center', color: s.color }}>
                <s.icon size={18} />
              </div>
            </div>
            <span className="kpi-value">{s.count}</span>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              {((s.count / reconciliationCases.length) * 100).toFixed(1)}% of total
            </span>
          </div>
        ))}
      </div>

      <div className="tabs">
        {(['all', 'matched', 'partial', 'exception'] as const).map((t) => (
          <div key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'all' ? 'All Cases' : t.charAt(0).toUpperCase() + t.slice(1)}
          </div>
        ))}
      </div>

      <div className="filter-bar">
        <div style={{ position: 'relative', flex: 1, maxWidth: 280 }}>
          <Search size={14} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-muted)' }} />
          <input
            type="text"
            placeholder="Search by case or transaction ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ paddingLeft: 30, width: '100%' }}
          />
        </div>
        <select value={caseFilter} onChange={(e) => setCaseFilter(e.target.value as CaseStatus | '')}>
          <option value="">All Case Statuses</option>
          <option value="open">Open</option>
          <option value="investigating">Investigating</option>
          <option value="resolved">Resolved</option>
          <option value="closed">Closed</option>
        </select>
      </div>

      <div className="card">
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Case ID</th>
                <th>Transaction</th>
                <th>Status</th>
                <th>Match Score</th>
                <th>Amount</th>
                <th>Expected</th>
                <th>Source → Target</th>
                <th>Case Status</th>
                <th>Updated</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.id}>
                  <td style={{ color: 'var(--accent-hover)', fontWeight: 600 }}>{r.id}</td>
                  <td style={{ fontWeight: 500 }}>{r.transactionId}</td>
                  <td><span className={`badge badge-${r.status}`}>{r.status}</span></td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div className="match-score-bar">
                        <div
                          className="match-score-fill"
                          style={{ width: `${r.matchScore}%`, background: getMatchColor(r.matchScore) }}
                        />
                      </div>
                      <span style={{ fontSize: 12, fontWeight: 600, color: getMatchColor(r.matchScore) }}>
                        {r.matchScore}%
                      </span>
                    </div>
                  </td>
                  <td className="amount">{formatCurrency(r.amount)}</td>
                  <td className="amount" style={{ color: r.amount !== r.expectedAmount ? 'var(--yellow)' : 'var(--text-secondary)' }}>
                    {formatCurrency(r.expectedAmount)}
                  </td>
                  <td style={{ fontSize: 12 }}>{r.source} → {r.target}</td>
                  <td><span className={`badge badge-${r.caseStatus}`}>{r.caseStatus}</span></td>
                  <td>{formatDateTime(r.updatedAt)}</td>
                  <td>
                    <div style={{ display: 'flex', gap: 4 }}>
                      <button className="btn btn-ghost btn-sm" title="Investigate"><FileText size={13} /></button>
                      <button className="btn btn-success btn-sm" title="Approve"><ThumbsUp size={13} /></button>
                      <button className="btn btn-danger btn-sm" title="Reject"><ThumbsDown size={13} /></button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-header">
          <div className="card-title">Reconciliation Log</div>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Case</th>
                <th>Action</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {reconciliationCases
                .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
                .map((r) => (
                  <tr key={`log-${r.id}`}>
                    <td>{formatDateTime(r.updatedAt)}</td>
                    <td style={{ fontWeight: 600 }}>{r.id}</td>
                    <td>
                      <span className={`badge badge-${r.caseStatus}`}>
                        {r.caseStatus === 'closed'
                          ? 'Auto-matched'
                          : r.caseStatus === 'investigating'
                          ? 'Under investigation'
                          : 'Pending review'}
                      </span>
                    </td>
                    <td style={{ fontSize: 12 }}>{r.notes}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
