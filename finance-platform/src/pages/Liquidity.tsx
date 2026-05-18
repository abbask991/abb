import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import {
  Landmark,
  CreditCard,
  Clock,
  Gift,
  Users,
  TrendingDown,
  ShieldCheck,
  AlertTriangle,
} from 'lucide-react';
import { bankBalances, pspBalances, transactions, liquidityTrend } from '../data/mockData';
import { formatCurrency, tooltipFormatter, tooltipPctFormatter } from '../components/FormatHelpers';

export default function Liquidity() {
  const totalBankBalance = bankBalances.reduce((s, b) => s + b.balance, 0);
  const totalPSPBalance = pspBalances.reduce((s, p) => s + p.balance, 0);
  const pendingWithdrawals = transactions
    .filter((t) => t.type === 'withdrawal' && t.status === 'pending')
    .reduce((s, t) => s + t.amount, 0);
  const totalPendingOut = pspBalances.reduce((s, p) => s + p.pendingOut, 0);

  const bonusExposure = 125000;
  const commissionLiabilities = 48500;

  const availableCash = totalBankBalance + totalPSPBalance;
  const netLiquidity = availableCash - pendingWithdrawals - bonusExposure - commissionLiabilities;
  const liquidityBuffer = ((netLiquidity / availableCash) * 100);

  const summaryKPIs = [
    { label: 'Available Cash', value: formatCurrency(availableCash), icon: Landmark, color: 'var(--green)', bg: 'var(--green-bg)' },
    { label: 'Net Liquidity', value: formatCurrency(netLiquidity), icon: ShieldCheck, color: 'var(--blue)', bg: 'var(--blue-bg)' },
    { label: 'Liquidity Buffer', value: `${liquidityBuffer.toFixed(1)}%`, icon: liquidityBuffer < 15 ? AlertTriangle : ShieldCheck, color: liquidityBuffer < 15 ? 'var(--red)' : 'var(--green)', bg: liquidityBuffer < 15 ? 'var(--red-bg)' : 'var(--green-bg)' },
    { label: 'Pending Withdrawals', value: formatCurrency(pendingWithdrawals + totalPendingOut), icon: Clock, color: 'var(--yellow)', bg: 'var(--yellow-bg)' },
  ];

  const liabilities = [
    { label: 'Pending Withdrawals', value: pendingWithdrawals + totalPendingOut, icon: TrendingDown, color: 'var(--red)' },
    { label: 'Bonus Exposure', value: bonusExposure, icon: Gift, color: 'var(--yellow)' },
    { label: 'Commission Liabilities', value: commissionLiabilities, icon: Users, color: 'var(--purple)' },
  ];

  const barData = [
    { name: 'Available Cash', value: availableCash, fill: '#22c55e' },
    { name: 'Withdrawals', value: pendingWithdrawals + totalPendingOut, fill: '#ef4444' },
    { name: 'Bonus Exposure', value: bonusExposure, fill: '#f59e0b' },
    { name: 'Commissions', value: commissionLiabilities, fill: '#a855f7' },
    { name: 'Net Liquidity', value: netLiquidity, fill: '#6366f1' },
  ];

  return (
    <>
      <div className="kpi-grid">
        {summaryKPIs.map((kpi) => (
          <div className="kpi-card" key={kpi.label}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span className="kpi-label">{kpi.label}</span>
              <div style={{ width: 34, height: 34, borderRadius: 8, background: kpi.bg, display: 'flex', alignItems: 'center', justifyContent: 'center', color: kpi.color }}>
                <kpi.icon size={18} />
              </div>
            </div>
            <span className="kpi-value">{kpi.value}</span>
          </div>
        ))}
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Bank Balances</div>
              <div className="card-subtitle">Real-time positions</div>
            </div>
          </div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Bank</th>
                  <th>Currency</th>
                  <th>Balance</th>
                  <th>Share</th>
                </tr>
              </thead>
              <tbody>
                {bankBalances.map((b) => (
                  <tr key={b.id}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Landmark size={14} style={{ color: 'var(--accent)' }} />
                        {b.bank}
                      </div>
                    </td>
                    <td>{b.currency}</td>
                    <td className="amount">{formatCurrency(b.balance, b.currency)}</td>
                    <td>
                      <div className="progress-bar" style={{ width: 80 }}>
                        <div className="progress-fill" style={{ width: `${(b.balance / totalBankBalance) * 100}%`, background: 'var(--accent)' }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">PSP Balances</div>
              <div className="card-subtitle">Payment service provider positions</div>
            </div>
          </div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>PSP</th>
                  <th>Currency</th>
                  <th>Balance</th>
                  <th>Pending In</th>
                  <th>Pending Out</th>
                </tr>
              </thead>
              <tbody>
                {pspBalances.map((p) => (
                  <tr key={p.id}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <CreditCard size={14} style={{ color: 'var(--cyan)' }} />
                        {p.psp}
                      </div>
                    </td>
                    <td>{p.currency}</td>
                    <td className="amount">{formatCurrency(p.balance, p.currency)}</td>
                    <td className="amount amount-positive">{p.pendingIn > 0 ? `+${formatCurrency(p.pendingIn, p.currency)}` : '–'}</td>
                    <td className="amount amount-negative">{p.pendingOut > 0 ? `-${formatCurrency(p.pendingOut, p.currency)}` : '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Liquidity Calculation</div>
              <div className="card-subtitle">Breakdown of available vs committed funds</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                type="number"
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={(v: number) => `$${(v / 1_000_000).toFixed(1)}M`}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                width={120}
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
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {barData.map((entry, index) => (
                  <rect key={index} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Liquidity Trend</div>
              <div className="card-subtitle">7-day buffer % and available cash</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={liquidityTrend}>
              <defs>
                <linearGradient id="bufferGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={(v: number) => `${v}%`} />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={tooltipPctFormatter}
              />
              <Area type="monotone" dataKey="buffer" stroke="#f59e0b" fill="url(#bufferGrad)" strokeWidth={2} name="Buffer %" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Liability Breakdown</div>
            <div className="card-subtitle">Committed funds and obligations</div>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {liabilities.map((l) => (
            <div key={l.label} style={{ padding: 16, background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <l.icon size={16} style={{ color: l.color }} />
                <span style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>{l.label}</span>
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, color: l.color }}>{formatCurrency(l.value)}</div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
