import { useState, useMemo } from 'react';
import { Search, Download, BookOpen, ArrowUpRight, ArrowDownLeft } from 'lucide-react';
import { ledgerEntries } from '../data/mockData';
import { formatCurrency, formatDateTime } from '../components/FormatHelpers';
import type { LedgerEntryType } from '../types';

const accountCategories = [
  { key: '', label: 'All Accounts' },
  { key: 'client_liability', label: 'Client Liability' },
  { key: 'company_cash', label: 'Company Cash' },
  { key: 'psp_accounts', label: 'PSP Accounts' },
  { key: 'fee_accounts', label: 'Fee Accounts' },
  { key: 'commission_accounts', label: 'Commission Accounts' },
];

export default function Ledger() {
  const [tab, setTab] = useState<'all' | LedgerEntryType>('all');
  const [accountFilter, setAccountFilter] = useState('');
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    return ledgerEntries.filter((e) => {
      if (tab !== 'all' && e.type !== tab) return false;
      if (accountFilter && e.accountCategory !== accountFilter) return false;
      if (search && !e.id.toLowerCase().includes(search.toLowerCase()) && !e.account.toLowerCase().includes(search.toLowerCase()) && !e.reference.toLowerCase().includes(search.toLowerCase()))
        return false;
      return true;
    });
  }, [tab, accountFilter, search]);

  const totalDebits = ledgerEntries.filter((e) => e.type === 'debit').reduce((s, e) => s + e.amount, 0);
  const totalCredits = ledgerEntries.filter((e) => e.type === 'credit').reduce((s, e) => s + e.amount, 0);

  const accountSummary = useMemo(() => {
    const map = new Map<string, { debit: number; credit: number; count: number }>();
    ledgerEntries.forEach((e) => {
      const curr = map.get(e.account) || { debit: 0, credit: 0, count: 0 };
      if (e.type === 'debit') curr.debit += e.amount;
      else curr.credit += e.amount;
      curr.count++;
      map.set(e.account, curr);
    });
    return Array.from(map.entries()).map(([name, data]) => ({
      name,
      ...data,
      net: data.debit - data.credit,
    }));
  }, []);

  return (
    <>
      <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <div className="kpi-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="kpi-label">Total Debits</span>
            <div style={{ width: 34, height: 34, borderRadius: 8, background: 'var(--cyan-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--cyan)' }}>
              <ArrowUpRight size={18} />
            </div>
          </div>
          <span className="kpi-value">{formatCurrency(totalDebits)}</span>
        </div>
        <div className="kpi-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="kpi-label">Total Credits</span>
            <div style={{ width: 34, height: 34, borderRadius: 8, background: 'var(--purple-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--purple)' }}>
              <ArrowDownLeft size={18} />
            </div>
          </div>
          <span className="kpi-value">{formatCurrency(totalCredits)}</span>
        </div>
        <div className="kpi-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="kpi-label">Entries</span>
            <div style={{ width: 34, height: 34, borderRadius: 8, background: 'var(--accent-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)' }}>
              <BookOpen size={18} />
            </div>
          </div>
          <span className="kpi-value">{ledgerEntries.length}</span>
        </div>
      </div>

      <div className="grid-2">
        <div>
          <div className="tabs">
            {(['all', 'debit', 'credit'] as const).map((t) => (
              <div key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
                {t === 'all' ? 'All Entries' : t === 'debit' ? 'Debit Entries' : 'Credit Entries'}
              </div>
            ))}
          </div>

          <div className="filter-bar">
            <div style={{ position: 'relative', flex: 1, maxWidth: 260 }}>
              <Search size={14} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="Search entries..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{ paddingLeft: 30, width: '100%' }}
              />
            </div>
            <select value={accountFilter} onChange={(e) => setAccountFilter(e.target.value)}>
              {accountCategories.map((c) => (
                <option key={c.key} value={c.key}>{c.label}</option>
              ))}
            </select>
            <button className="btn btn-ghost btn-sm"><Download size={14} /> Export</button>
          </div>

          <div className="card">
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Account</th>
                    <th>Amount</th>
                    <th>Reference</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((e) => (
                    <tr key={e.id}>
                      <td style={{ color: 'var(--accent-hover)', fontWeight: 600 }}>{e.id}</td>
                      <td><span className={`badge badge-${e.type}`}>{e.type}</span></td>
                      <td style={{ fontSize: 12 }}>{e.account}</td>
                      <td className="amount">{formatCurrency(e.amount, e.currency)}</td>
                      <td style={{ fontWeight: 500 }}>{e.reference}</td>
                      <td>{formatDateTime(e.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div>
          <div className="card">
            <div className="card-header">
              <div className="card-title">Account Summary</div>
            </div>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Account</th>
                    <th>Debits</th>
                    <th>Credits</th>
                    <th>Net</th>
                    <th>Entries</th>
                  </tr>
                </thead>
                <tbody>
                  {accountSummary.map((a) => (
                    <tr key={a.name}>
                      <td style={{ fontSize: 12, fontWeight: 500 }}>{a.name}</td>
                      <td className="amount" style={{ color: 'var(--cyan)' }}>{formatCurrency(a.debit)}</td>
                      <td className="amount" style={{ color: 'var(--purple)' }}>{formatCurrency(a.credit)}</td>
                      <td className={`amount ${a.net >= 0 ? 'amount-positive' : 'amount-negative'}`}>{formatCurrency(Math.abs(a.net))}</td>
                      <td>{a.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-header">
              <div className="card-title">Audit Trail</div>
              <div className="card-subtitle">Recent ledger entries</div>
            </div>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Entry</th>
                    <th>Action</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {ledgerEntries
                    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
                    .slice(0, 8)
                    .map((e) => (
                      <tr key={`audit-${e.id}`}>
                        <td>{formatDateTime(e.timestamp)}</td>
                        <td style={{ fontWeight: 600 }}>{e.id}</td>
                        <td><span className={`badge badge-${e.type}`}>{e.type === 'debit' ? 'DR' : 'CR'}</span></td>
                        <td style={{ fontSize: 12 }}>{e.description}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
