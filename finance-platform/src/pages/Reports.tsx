import { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from 'recharts';
import {
  FileText,
  Calendar,
  TrendingUp,
  Download,
  DollarSign,
  Users,
  Megaphone,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react';
import { profitabilityData, ibProfitabilityData, monthlyKPIs, cashFlowData } from '../data/mockData';
import { formatCurrency, tooltipFormatter } from '../components/FormatHelpers';

const reportTypes = [
  { id: 'daily', label: 'Daily Finance Report', icon: Calendar, description: 'End-of-day financial summary with all transaction activity and reconciliation status.' },
  { id: 'weekly', label: 'Weekly Control Report', icon: FileText, description: 'Weekly compliance and control metrics with exception analysis.' },
  { id: 'monthly', label: 'Monthly Financial Report', icon: TrendingUp, description: 'Comprehensive monthly financial performance report with P&L analysis.' },
];

export default function Reports() {
  const [tab, setTab] = useState<'profitability' | 'kpis' | 'reports'>('profitability');
  const [profitTab, setProfitTab] = useState<'client' | 'ib' | 'campaign'>('client');

  const totalRevenue = profitabilityData.reduce((s, d) => s + d.revenue, 0);
  const totalCosts = profitabilityData.reduce((s, d) => s + d.costs, 0);
  const totalProfit = profitabilityData.reduce((s, d) => s + d.profit, 0);
  const margin = ((totalProfit / totalRevenue) * 100).toFixed(1);

  const campaignData = [
    { name: 'Welcome Bonus', spend: 45000, deposits: 180000, roi: 300 },
    { name: 'Loyalty Program', spend: 22000, deposits: 95000, roi: 332 },
    { name: 'Referral Bonus', spend: 15000, deposits: 72000, roi: 380 },
    { name: 'VIP Cashback', spend: 35000, deposits: 110000, roi: 214 },
  ];

  return (
    <>
      <div className="tabs">
        {([
          { id: 'profitability' as const, label: 'Profitability Analysis' },
          { id: 'kpis' as const, label: 'Financial KPIs' },
          { id: 'reports' as const, label: 'Generated Reports' },
        ]).map((t) => (
          <div key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
            {t.label}
          </div>
        ))}
      </div>

      {tab === 'profitability' && (
        <>
          <div className="kpi-grid">
            <div className="kpi-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="kpi-label">Total Revenue</span>
                <div style={{ width: 34, height: 34, borderRadius: 8, background: 'var(--green-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--green)' }}>
                  <DollarSign size={18} />
                </div>
              </div>
              <span className="kpi-value">{formatCurrency(totalRevenue)}</span>
              <span className="kpi-change up"><ArrowUpRight size={14} /> 8.2%</span>
            </div>
            <div className="kpi-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="kpi-label">Operating Costs</span>
                <div style={{ width: 34, height: 34, borderRadius: 8, background: 'var(--red-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--red)' }}>
                  <ArrowDownRight size={18} />
                </div>
              </div>
              <span className="kpi-value">{formatCurrency(totalCosts)}</span>
              <span className="kpi-change down"><ArrowDownRight size={14} /> 3.1%</span>
            </div>
            <div className="kpi-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="kpi-label">Net Profit</span>
                <div style={{ width: 34, height: 34, borderRadius: 8, background: 'var(--accent-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)' }}>
                  <TrendingUp size={18} />
                </div>
              </div>
              <span className="kpi-value">{formatCurrency(totalProfit)}</span>
              <span className="kpi-change up"><ArrowUpRight size={14} /> 12.4%</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Profit Margin</span>
              <span className="kpi-value">{margin}%</span>
              <span className="kpi-change up"><ArrowUpRight size={14} /> 1.8pp</span>
            </div>
          </div>

          <div className="tabs" style={{ marginBottom: 16 }}>
            {([
              { id: 'client' as const, label: 'Client Profitability', icon: Users },
              { id: 'ib' as const, label: 'IB Profitability', icon: Users },
              { id: 'campaign' as const, label: 'Campaign Profitability', icon: Megaphone },
            ]).map((t) => (
              <div key={t.id} className={`tab ${profitTab === t.id ? 'active' : ''}`} onClick={() => setProfitTab(t.id)}>
                {t.label}
              </div>
            ))}
          </div>

          {profitTab === 'client' && (
            <div className="grid-2">
              <div className="card">
                <div className="card-header">
                  <div className="card-title">Client Revenue vs Costs</div>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={profitabilityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                    <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} formatter={tooltipFormatter} />
                    <Bar dataKey="revenue" fill="#22c55e" radius={[4, 4, 0, 0]} name="Revenue" />
                    <Bar dataKey="costs" fill="#ef4444" radius={[4, 4, 0, 0]} name="Costs" />
                    <Bar dataKey="profit" fill="#6366f1" radius={[4, 4, 0, 0]} name="Profit" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="card">
                <div className="card-header">
                  <div className="card-title">Profit Breakdown</div>
                </div>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr><th>Client</th><th>Revenue</th><th>Costs</th><th>Profit</th><th>Margin</th></tr>
                    </thead>
                    <tbody>
                      {profitabilityData.sort((a, b) => b.profit - a.profit).map((d) => (
                        <tr key={d.name}>
                          <td style={{ fontWeight: 600 }}>{d.name}</td>
                          <td className="amount amount-positive">{formatCurrency(d.revenue)}</td>
                          <td className="amount amount-negative">{formatCurrency(d.costs)}</td>
                          <td className="amount" style={{ color: 'var(--accent-hover)' }}>{formatCurrency(d.profit)}</td>
                          <td>{((d.profit / d.revenue) * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {profitTab === 'ib' && (
            <div className="grid-2">
              <div className="card">
                <div className="card-header">
                  <div className="card-title">IB Partner Performance</div>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={ibProfitabilityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                    <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} formatter={tooltipFormatter} />
                    <Bar dataKey="netRevenue" fill="#22c55e" radius={[4, 4, 0, 0]} name="Net Revenue" />
                    <Bar dataKey="commission" fill="#a855f7" radius={[4, 4, 0, 0]} name="Commission" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="card">
                <div className="card-header">
                  <div className="card-title">IB Details</div>
                </div>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr><th>Partner</th><th>Clients</th><th>Volume</th><th>Commission</th><th>Net Rev</th></tr>
                    </thead>
                    <tbody>
                      {ibProfitabilityData.sort((a, b) => b.netRevenue - a.netRevenue).map((d) => (
                        <tr key={d.name}>
                          <td style={{ fontWeight: 600 }}>{d.name}</td>
                          <td>{d.clients}</td>
                          <td className="amount">{formatCurrency(d.volume)}</td>
                          <td className="amount" style={{ color: 'var(--purple)' }}>{formatCurrency(d.commission)}</td>
                          <td className="amount amount-positive">{formatCurrency(d.netRevenue)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {profitTab === 'campaign' && (
            <div className="grid-2">
              <div className="card">
                <div className="card-header">
                  <div className="card-title">Campaign ROI</div>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={campaignData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                    <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} formatter={tooltipFormatter} />
                    <Bar dataKey="spend" fill="#ef4444" radius={[4, 4, 0, 0]} name="Spend" />
                    <Bar dataKey="deposits" fill="#22c55e" radius={[4, 4, 0, 0]} name="Deposits Generated" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="card">
                <div className="card-header">
                  <div className="card-title">Campaign Performance</div>
                </div>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr><th>Campaign</th><th>Spend</th><th>Deposits</th><th>ROI</th></tr>
                    </thead>
                    <tbody>
                      {campaignData.sort((a, b) => b.roi - a.roi).map((d) => (
                        <tr key={d.name}>
                          <td style={{ fontWeight: 600 }}>{d.name}</td>
                          <td className="amount amount-negative">{formatCurrency(d.spend)}</td>
                          <td className="amount amount-positive">{formatCurrency(d.deposits)}</td>
                          <td>
                            <span className={`badge ${d.roi > 300 ? 'badge-completed' : 'badge-pending'}`}>{d.roi}%</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {tab === 'kpis' && (
        <>
          <div className="grid-2">
            <div className="card">
              <div className="card-header">
                <div className="card-title">Net Flow Trend</div>
                <div className="card-subtitle">Monthly net cash flow</div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={monthlyKPIs}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="month" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={(v: number) => `$${(v / 1_000_000).toFixed(1)}M`} />
                  <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} formatter={tooltipFormatter} />
                  <Line type="monotone" dataKey="netFlow" stroke="#6366f1" strokeWidth={2.5} dot={{ fill: '#6366f1', r: 4 }} name="Net Flow" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-title">Operating Costs</div>
                <div className="card-subtitle">Monthly operating expenses</div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={monthlyKPIs}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="month" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                  <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} formatter={tooltipFormatter} />
                  <Bar dataKey="opCosts" fill="#ef4444" radius={[4, 4, 0, 0]} name="Operating Costs" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid-2">
            <div className="card">
              <div className="card-header">
                <div className="card-title">Exception Trends</div>
                <div className="card-subtitle">Monthly exception count</div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={monthlyKPIs}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="month" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} />
                  <Line type="monotone" dataKey="exceptions" stroke="#f59e0b" strokeWidth={2.5} dot={{ fill: '#f59e0b', r: 4 }} name="Exceptions" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-title">Daily Cash Flow (Current Week)</div>
                <div className="card-subtitle">Deposits, withdrawals, and net flow</div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={cashFlowData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                  <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }} formatter={tooltipFormatter} />
                  <Legend />
                  <Bar dataKey="deposits" fill="#22c55e" radius={[4, 4, 0, 0]} name="Deposits" />
                  <Bar dataKey="withdrawals" fill="#ef4444" radius={[4, 4, 0, 0]} name="Withdrawals" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div className="card-title">Monthly KPI Summary</div>
            </div>
            <div className="table-container">
              <table>
                <thead>
                  <tr><th>Month</th><th>Net Flow</th><th>Operating Costs</th><th>Exceptions</th><th>Cost Ratio</th></tr>
                </thead>
                <tbody>
                  {monthlyKPIs.map((m) => (
                    <tr key={m.month}>
                      <td style={{ fontWeight: 600 }}>{m.month} 2026</td>
                      <td className="amount amount-positive">{formatCurrency(m.netFlow)}</td>
                      <td className="amount amount-negative">{formatCurrency(m.opCosts)}</td>
                      <td>
                        <span className={`badge ${m.exceptions > 12 ? 'badge-failed' : m.exceptions > 8 ? 'badge-pending' : 'badge-completed'}`}>
                          {m.exceptions}
                        </span>
                      </td>
                      <td>{((m.opCosts / m.netFlow) * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {tab === 'reports' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {reportTypes.map((report) => (
            <div className="card" key={report.id} style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <div style={{ width: 48, height: 48, borderRadius: 10, background: 'var(--accent-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)', flexShrink: 0 }}>
                <report.icon size={22} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>{report.label}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5 }}>{report.description}</div>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-primary btn-sm"><Download size={13} /> Generate</button>
                <button className="btn btn-ghost btn-sm">Schedule</button>
              </div>
            </div>
          ))}

          <div className="card" style={{ marginTop: 8 }}>
            <div className="card-header">
              <div className="card-title">Recent Reports</div>
            </div>
            <div className="table-container">
              <table>
                <thead>
                  <tr><th>Report</th><th>Period</th><th>Generated</th><th>Status</th><th></th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Daily Finance Report</td>
                    <td>May 17, 2026</td>
                    <td>May 18, 2026 01:00</td>
                    <td><span className="badge badge-completed">Ready</span></td>
                    <td><button className="btn btn-ghost btn-sm"><Download size={13} /></button></td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Weekly Control Report</td>
                    <td>May 12–18, 2026</td>
                    <td>May 18, 2026 06:00</td>
                    <td><span className="badge badge-completed">Ready</span></td>
                    <td><button className="btn btn-ghost btn-sm"><Download size={13} /></button></td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Monthly Financial Report</td>
                    <td>April 2026</td>
                    <td>May 01, 2026 08:00</td>
                    <td><span className="badge badge-completed">Ready</span></td>
                    <td><button className="btn btn-ghost btn-sm"><Download size={13} /></button></td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Daily Finance Report</td>
                    <td>May 18, 2026</td>
                    <td>—</td>
                    <td><span className="badge badge-pending">Scheduled 01:00</span></td>
                    <td></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
