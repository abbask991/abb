import { useState, useMemo } from 'react';
import { Search, Download, Eye, X } from 'lucide-react';
import { transactions } from '../data/mockData';
import { formatCurrency, formatDateTime } from '../components/FormatHelpers';
import type { Transaction, TransactionType, TransactionStatus } from '../types';

export default function Transactions() {
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<TransactionType | ''>('');
  const [statusFilter, setStatusFilter] = useState<TransactionStatus | ''>('');
  const [pspFilter, setPspFilter] = useState('');
  const [bankFilter, setBankFilter] = useState('');
  const [selected, setSelected] = useState<Transaction | null>(null);

  const psps = useMemo(() => [...new Set(transactions.map((t) => t.psp))], []);
  const banks = useMemo(() => [...new Set(transactions.map((t) => t.bank))], []);

  const filtered = useMemo(() => {
    return transactions.filter((t) => {
      if (search && !t.id.toLowerCase().includes(search.toLowerCase()) && !t.client.toLowerCase().includes(search.toLowerCase()) && !t.description.toLowerCase().includes(search.toLowerCase()))
        return false;
      if (typeFilter && t.type !== typeFilter) return false;
      if (statusFilter && t.status !== statusFilter) return false;
      if (pspFilter && t.psp !== pspFilter) return false;
      if (bankFilter && t.bank !== bankFilter) return false;
      return true;
    });
  }, [search, typeFilter, statusFilter, pspFilter, bankFilter]);

  return (
    <>
      <div className="section-header">
        <div>
          <div className="section-title">All Transactions</div>
          <div className="section-subtitle">{filtered.length} of {transactions.length} transactions</div>
        </div>
        <button className="btn btn-ghost btn-sm">
          <Download size={14} /> Export
        </button>
      </div>

      <div className="filter-bar">
        <div style={{ position: 'relative', flex: 1, maxWidth: 280 }}>
          <Search size={14} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-muted)' }} />
          <input
            type="text"
            placeholder="Search ID, client, description..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ paddingLeft: 30, width: '100%' }}
          />
        </div>
        <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value as TransactionType | '')}>
          <option value="">All Types</option>
          <option value="deposit">Deposits</option>
          <option value="withdrawal">Withdrawals</option>
          <option value="transfer">Transfers</option>
          <option value="fee">Fees</option>
          <option value="commission">Commissions</option>
        </select>
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as TransactionStatus | '')}>
          <option value="">All Statuses</option>
          <option value="completed">Completed</option>
          <option value="pending">Pending</option>
          <option value="failed">Failed</option>
          <option value="reversed">Reversed</option>
        </select>
        <select value={pspFilter} onChange={(e) => setPspFilter(e.target.value)}>
          <option value="">All PSPs</option>
          {psps.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <select value={bankFilter} onChange={(e) => setBankFilter(e.target.value)}>
          <option value="">All Banks</option>
          {banks.map((b) => <option key={b} value={b}>{b}</option>)}
        </select>
      </div>

      <div className="card">
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Type</th>
                <th>Client</th>
                <th>Amount</th>
                <th>PSP</th>
                <th>Bank</th>
                <th>Status</th>
                <th>Timestamp</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((tx) => (
                <tr key={tx.id} style={{ cursor: 'pointer' }} onClick={() => setSelected(tx)}>
                  <td style={{ color: 'var(--accent-hover)', fontWeight: 600 }}>{tx.id}</td>
                  <td><span className={`badge badge-${tx.type}`}>{tx.type}</span></td>
                  <td>{tx.client}</td>
                  <td>
                    <span className={`amount ${tx.type === 'deposit' ? 'amount-positive' : tx.type === 'withdrawal' ? 'amount-negative' : 'amount-neutral'}`}>
                      {tx.type === 'withdrawal' ? '-' : ''}{formatCurrency(tx.amount, tx.currency)}
                    </span>
                  </td>
                  <td>{tx.psp}</td>
                  <td>{tx.bank}</td>
                  <td><span className={`badge badge-${tx.status}`}>{tx.status}</span></td>
                  <td>{formatDateTime(tx.timestamp)}</td>
                  <td><Eye size={14} style={{ color: 'var(--text-muted)' }} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selected && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
          }}
          onClick={() => setSelected(null)}
        >
          <div
            className="card"
            style={{ width: 480, maxHeight: '80vh', overflow: 'auto' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="card-header">
              <div className="card-title">Transaction Details</div>
              <button className="btn btn-ghost btn-sm" onClick={() => setSelected(null)}>
                <X size={14} />
              </button>
            </div>
            <div className="detail-row">
              <span className="detail-label">ID</span>
              <span className="detail-value">{selected.id}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Type</span>
              <span className="detail-value"><span className={`badge badge-${selected.type}`}>{selected.type}</span></span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Amount</span>
              <span className="detail-value amount">{formatCurrency(selected.amount, selected.currency)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Currency</span>
              <span className="detail-value">{selected.currency}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Status</span>
              <span className="detail-value"><span className={`badge badge-${selected.status}`}>{selected.status}</span></span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Client</span>
              <span className="detail-value">{selected.client}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Source System</span>
              <span className="detail-value">{selected.sourceSystem}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">PSP</span>
              <span className="detail-value">{selected.psp}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Bank</span>
              <span className="detail-value">{selected.bank}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Timestamp</span>
              <span className="detail-value">{formatDateTime(selected.timestamp)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Description</span>
              <span className="detail-value">{selected.description}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Linked Records</span>
              <span className="detail-value">
                {selected.linkedRecords.length > 0
                  ? selected.linkedRecords.join(', ')
                  : 'None'}
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
