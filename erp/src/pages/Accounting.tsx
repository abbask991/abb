import { useState } from 'react';
import { Plus, Download, DollarSign, CreditCard, FileText, AlertTriangle } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts';
import PageHeader from '../components/PageHeader';
import Tabs from '../components/Tabs';
import StatusBadge from '../components/StatusBadge';
import KpiCard from '../components/KpiCard';
import { formatCurrency, formatCurrencyExact, formatDate, tooltipCurrencyFormatter } from '../components/FormatHelpers';
import { accounts, journalEntries, invoices, revenueByMonth } from '../data/mockData';

const accountTypeData = ['asset', 'liability', 'equity', 'revenue', 'expense'].map((type) => ({
  type: type.charAt(0).toUpperCase() + type.slice(1),
  balance: Math.abs(accounts.filter((a) => a.type === type).reduce((s, a) => s + a.balance, 0)),
}));

const TYPE_COLORS: Record<string, string> = {
  Asset: '#2563eb',
  Liability: '#dc2626',
  Equity: '#059669',
  Revenue: '#7c3aed',
  Expense: '#d97706',
};

const invoiceStatusData = ['paid', 'sent', 'overdue', 'draft'].map((s) => ({
  name: s.charAt(0).toUpperCase() + s.slice(1),
  value: invoices.filter((i) => i.status === s).length,
}));
const INV_COLORS = ['#059669', '#2563eb', '#dc2626', '#94a3b8'];

const tabs = [
  { id: 'overview', label: 'Overview' },
  { id: 'accounts', label: 'Chart of Accounts', count: accounts.length },
  { id: 'journal', label: 'Journal Entries', count: journalEntries.length },
  { id: 'invoices', label: 'Invoices', count: invoices.length },
];

export default function Accounting() {
  const [activeTab, setActiveTab] = useState('overview');

  const totalAssets = accounts.filter((a) => a.type === 'asset').reduce((s, a) => s + a.balance, 0);
  const totalLiabilities = accounts.filter((a) => a.type === 'liability').reduce((s, a) => s + a.balance, 0);
  const totalRevenue = accounts.filter((a) => a.type === 'revenue').reduce((s, a) => s + a.balance, 0);
  const totalExpenses = accounts.filter((a) => a.type === 'expense').reduce((s, a) => s + a.balance, 0);
  const arBalance = accounts.find((a) => a.code === '1100')?.balance ?? 0;
  const apBalance = accounts.find((a) => a.code === '2000')?.balance ?? 0;

  return (
    <div>
      <PageHeader
        title="Accounting"
        subtitle="Financial management, chart of accounts, journal entries, and invoicing."
        actions={
          <>
            <button className="btn-secondary"><Download className="w-4 h-4" /> Export</button>
            <button className="btn-primary"><Plus className="w-4 h-4" /> New Entry</button>
          </>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard title="Total Assets" value={formatCurrency(totalAssets)} icon={DollarSign} color="blue" />
        <KpiCard title="Accounts Receivable" value={formatCurrency(arBalance)} icon={CreditCard} color="emerald" />
        <KpiCard title="Accounts Payable" value={formatCurrency(apBalance)} icon={FileText} color="amber" />
        <KpiCard title="Net Income (YTD)" value={formatCurrency(totalRevenue - totalExpenses)} change="+18.4%" trend="up" icon={DollarSign} color="purple" />
      </div>

      <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />

      {activeTab === 'overview' && (
        <div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div className="card">
              <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Balances by Account Type</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={accountTypeData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="type" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v / 1000}k`} />
                    <Tooltip formatter={tooltipCurrencyFormatter} />
                    <Bar dataKey="balance" radius={[4, 4, 0, 0]} barSize={40} name="Balance">
                      {accountTypeData.map((entry) => (
                        <Cell key={entry.type} fill={TYPE_COLORS[entry.type]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card">
              <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Invoice Status</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie data={invoiceStatusData} cx="50%" cy="50%" innerRadius={55} outerRadius={90} paddingAngle={3} dataKey="value" nameKey="name">
                      {invoiceStatusData.map((_, i) => <Cell key={i} fill={INV_COLORS[i]} />)}
                    </Pie>
                    <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Profit & Loss Summary (YTD)</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={revenueByMonth.filter((m) => m.revenue > 0)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v / 1000}k`} />
                    <Tooltip formatter={tooltipCurrencyFormatter} />
                    <Bar dataKey="profit" fill="#059669" radius={[4, 4, 0, 0]} barSize={32} name="Net Profit" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card">
              <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Balance Sheet Summary</h3></div>
              <div className="card-body space-y-3">
                <div className="p-3 bg-blue-50 rounded-lg flex justify-between items-center">
                  <span className="text-sm font-medium text-blue-800">Total Assets</span>
                  <span className="font-bold text-blue-600">{formatCurrency(totalAssets)}</span>
                </div>
                <div className="p-3 bg-red-50 rounded-lg flex justify-between items-center">
                  <span className="text-sm font-medium text-red-800">Total Liabilities</span>
                  <span className="font-bold text-red-600">{formatCurrency(totalLiabilities)}</span>
                </div>
                <div className="p-3 bg-emerald-50 rounded-lg flex justify-between items-center">
                  <span className="text-sm font-medium text-emerald-800">Total Equity</span>
                  <span className="font-bold text-emerald-600">{formatCurrency(totalAssets - totalLiabilities)}</span>
                </div>
                <div className="border-t pt-3 flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-600">Liabilities + Equity</span>
                  <span className="font-bold text-gray-900">{formatCurrency(totalLiabilities + (totalAssets - totalLiabilities))}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'accounts' && (
        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Chart of Accounts</h3></div>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Code</th>
                  <th>Account Name</th>
                  <th>Type</th>
                  <th>Balance</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {accounts.map((a) => (
                  <tr key={a.id}>
                    <td className="font-mono text-sm font-medium">{a.code}</td>
                    <td className="font-medium">{a.name}</td>
                    <td>
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium" style={{ backgroundColor: `${TYPE_COLORS[a.type.charAt(0).toUpperCase() + a.type.slice(1)]}15`, color: TYPE_COLORS[a.type.charAt(0).toUpperCase() + a.type.slice(1)] }}>
                        {a.type.charAt(0).toUpperCase() + a.type.slice(1)}
                      </span>
                    </td>
                    <td className={`font-bold ${a.balance < 0 ? 'text-red-600' : ''}`}>{formatCurrency(a.balance)}</td>
                    <td><StatusBadge status={a.isActive ? 'active' : 'inactive'} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'journal' && (
        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Journal Entries</h3></div>
          <div className="space-y-4 p-6">
            {journalEntries.map((je) => (
              <div key={je.id} className="border border-gray-200 rounded-lg overflow-hidden">
                <div className="flex items-center justify-between p-4 bg-gray-50">
                  <div className="flex items-center gap-4">
                    <span className="font-mono text-sm font-medium">{je.entryNumber}</span>
                    <span className="text-sm text-gray-500">{formatDate(je.date)}</span>
                    <span className="text-sm text-gray-700">{je.description}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-500">by {je.createdBy}</span>
                    <StatusBadge status={je.status} />
                  </div>
                </div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left px-4 py-2 text-xs text-gray-500 font-medium">Account</th>
                      <th className="text-right px-4 py-2 text-xs text-gray-500 font-medium">Debit</th>
                      <th className="text-right px-4 py-2 text-xs text-gray-500 font-medium">Credit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {je.lines.map((line, i) => (
                      <tr key={i} className="border-b border-gray-50">
                        <td className="px-4 py-2">{line.accountName}</td>
                        <td className="px-4 py-2 text-right font-medium">{line.debit > 0 ? formatCurrencyExact(line.debit) : ''}</td>
                        <td className="px-4 py-2 text-right font-medium">{line.credit > 0 ? formatCurrencyExact(line.credit) : ''}</td>
                      </tr>
                    ))}
                    <tr className="bg-gray-50 font-bold">
                      <td className="px-4 py-2">Total</td>
                      <td className="px-4 py-2 text-right">{formatCurrencyExact(je.lines.reduce((s, l) => s + l.debit, 0))}</td>
                      <td className="px-4 py-2 text-right">{formatCurrencyExact(je.lines.reduce((s, l) => s + l.credit, 0))}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'invoices' && (
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Invoices</h3>
            <button className="btn-primary text-sm"><Plus className="w-4 h-4" /> New Invoice</button>
          </div>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Invoice #</th>
                  <th>Customer</th>
                  <th>Issue Date</th>
                  <th>Due Date</th>
                  <th>Total</th>
                  <th>Paid</th>
                  <th>Balance</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {invoices.map((inv) => (
                  <tr key={inv.id}>
                    <td className="font-mono text-sm font-medium">{inv.invoiceNumber}</td>
                    <td className="font-medium">{inv.customerName}</td>
                    <td>{formatDate(inv.issueDate)}</td>
                    <td>{formatDate(inv.dueDate)}</td>
                    <td className="font-bold">{formatCurrencyExact(inv.total)}</td>
                    <td className="text-emerald-600">{formatCurrencyExact(inv.amountPaid)}</td>
                    <td className={inv.total - inv.amountPaid > 0 ? 'text-red-600 font-medium' : ''}>
                      {formatCurrencyExact(inv.total - inv.amountPaid)}
                    </td>
                    <td>
                      <div className="flex items-center gap-1">
                        {inv.status === 'overdue' && <AlertTriangle className="w-3 h-3 text-red-500" />}
                        <StatusBadge status={inv.status} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
