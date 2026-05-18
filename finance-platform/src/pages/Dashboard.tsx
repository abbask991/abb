import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  DollarSign,
  ArrowDownUp,
  Droplets,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react';
import { transactions, alerts, cashFlowData, liquidityTrend, bankBalances } from '../data/mockData';
import { formatCurrency, tooltipFormatter } from '../components/FormatHelpers';

const COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#3b82f6', '#a855f7'];

export default function Dashboard() {
  const totalDeposits = transactions
    .filter((t) => t.type === 'deposit' && t.status === 'completed')
    .reduce((s, t) => s + t.amount, 0);
  const totalWithdrawals = transactions
    .filter((t) => t.type === 'withdrawal' && t.status === 'completed')
    .reduce((s, t) => s + t.amount, 0);
  const netFlowToday = totalDeposits - totalWithdrawals;
  const totalCash = bankBalances.reduce((s, b) => s + b.balance, 0);
  const openAlerts = alerts.filter((a) => a.status === 'open' || a.status === 'investigating');
  const criticalAlerts = alerts.filter((a) => a.severity === 'critical' && a.status !== 'resolved');
  const currentBuffer = liquidityTrend[liquidityTrend.length - 1].buffer;

  const txTypeBreakdown = [
    { name: 'Deposits', value: transactions.filter((t) => t.type === 'deposit').length },
    { name: 'Withdrawals', value: transactions.filter((t) => t.type === 'withdrawal').length },
    { name: 'Transfers', value: transactions.filter((t) => t.type === 'transfer').length },
    { name: 'Fees', value: transactions.filter((t) => t.type === 'fee').length },
    { name: 'Commissions', value: transactions.filter((t) => t.type === 'commission').length },
  ];

  const kpis = [
    {
      label: 'Cash Position',
      value: formatCurrency(totalCash),
      change: 3.2,
      trend: 'up' as const,
      icon: DollarSign,
      color: 'var(--green)',
      bg: 'var(--green-bg)',
    },
    {
      label: 'Net Flow Today',
      value: formatCurrency(netFlowToday),
      change: 12.5,
      trend: 'up' as const,
      icon: ArrowDownUp,
      color: 'var(--blue)',
      bg: 'var(--blue-bg)',
    },
    {
      label: 'Liquidity Buffer',
      value: `${currentBuffer}%`,
      change: -2.6,
      trend: 'down' as const,
      icon: Droplets,
      color: currentBuffer < 15 ? 'var(--red)' : 'var(--green)',
      bg: currentBuffer < 15 ? 'var(--red-bg)' : 'var(--green-bg)',
    },
    {
      label: 'Active Alerts',
      value: openAlerts.length,
      change: criticalAlerts.length,
      trend: criticalAlerts.length > 0 ? ('down' as const) : ('flat' as const),
      icon: AlertTriangle,
      color: 'var(--yellow)',
      bg: 'var(--yellow-bg)',
    },
  ];

  return (
    <>
      <div className="kpi-grid">
        {kpis.map((kpi) => (
          <div className="kpi-card" key={kpi.label}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span className="kpi-label">{kpi.label}</span>
              <div
                style={{
                  width: 34,
                  height: 34,
                  borderRadius: 8,
                  background: kpi.bg,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: kpi.color,
                }}
              >
                <kpi.icon size={18} />
              </div>
            </div>
            <span className="kpi-value">{kpi.value}</span>
            <span className={`kpi-change ${kpi.trend}`}>
              {kpi.trend === 'up' ? (
                <ArrowUpRight size={14} />
              ) : kpi.trend === 'down' ? (
                <ArrowDownRight size={14} />
              ) : null}
              {kpi.trend === 'down' && kpi.label === 'Active Alerts'
                ? `${kpi.change} critical`
                : `${Math.abs(kpi.change)}%`}
            </span>
          </div>
        ))}
      </div>

      <div className="grid-2-1">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Cash Flow (7-Day)</div>
              <div className="card-subtitle">Deposits vs Withdrawals</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={cashFlowData}>
              <defs>
                <linearGradient id="colorDeposits" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorWithdrawals" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={tooltipFormatter}
              />
              <Area
                type="monotone"
                dataKey="deposits"
                stroke="#22c55e"
                fill="url(#colorDeposits)"
                strokeWidth={2}
                name="Deposits"
              />
              <Area
                type="monotone"
                dataKey="withdrawals"
                stroke="#ef4444"
                fill="url(#colorWithdrawals)"
                strokeWidth={2}
                name="Withdrawals"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Transaction Mix</div>
              <div className="card-subtitle">By type</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={txTypeBreakdown}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={75}
                paddingAngle={3}
                dataKey="value"
              >
                {txTypeBreakdown.map((_, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }}>
            {txTypeBreakdown.map((item, i) => (
              <div
                key={item.name}
                style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--text-secondary)' }}
              >
                <div style={{ width: 8, height: 8, borderRadius: 2, background: COLORS[i] }} />
                {item.name} ({item.value})
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Liquidity Trend</div>
              <div className="card-subtitle">Available cash & buffer %</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={liquidityTrend}>
              <defs>
                <linearGradient id="colorLiquidity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={(v: number) => `$${(v / 1_000_000).toFixed(1)}M`}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={tooltipFormatter}
              />
              <Area
                type="monotone"
                dataKey="available"
                stroke="#6366f1"
                fill="url(#colorLiquidity)"
                strokeWidth={2}
                name="Available Cash"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Net Flow (7-Day)</div>
              <div className="card-subtitle">Daily net cash flow</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={cashFlowData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={tooltipFormatter}
              />
              <Bar dataKey="netFlow" fill="#6366f1" radius={[4, 4, 0, 0]} name="Net Flow" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Active Alerts</div>
            <div className="card-subtitle">{openAlerts.length} alerts requiring attention</div>
          </div>
        </div>
        {openAlerts.map((alert) => (
          <div className="alert-item" key={alert.id}>
            <div className={`alert-icon ${alert.severity}`}>
              <AlertTriangle size={18} />
            </div>
            <div className="alert-body">
              <div className="alert-title">{alert.title}</div>
              <div className="alert-desc">{alert.description}</div>
              <div className="alert-meta">
                <span className={`badge badge-${alert.severity}`}>{alert.severity}</span>
                <span className={`badge badge-${alert.category}`}>{alert.category}</span>
                <span className={`badge badge-${alert.status}`}>{alert.status}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
