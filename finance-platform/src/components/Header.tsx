import { useLocation } from 'react-router-dom';
import { Bell, Settings } from 'lucide-react';
import { alerts } from '../data/mockData';

const titles: Record<string, string> = {
  '/': 'Dashboard',
  '/transactions': 'Transactions',
  '/reconciliation': 'Reconciliation',
  '/ledger': 'Ledger',
  '/liquidity': 'Liquidity',
  '/alerts': 'Alerts & Exceptions',
  '/reports': 'Reports & Analytics',
};

export default function Header() {
  const location = useLocation();
  const title = titles[location.pathname] || 'Finance Platform';
  const openAlerts = alerts.filter((a) => a.status === 'open').length;

  return (
    <header className="header">
      <div className="header-left">
        <h2>{title}</h2>
        <span className="header-breadcrumb">Finance Automation Platform / {title}</span>
      </div>
      <div className="header-right">
        <div className="header-status">
          <span className="dot" />
          System Online
        </div>
        <span className="header-time">
          {new Date().toLocaleDateString('en-US', {
            weekday: 'short',
            month: 'short',
            day: 'numeric',
            year: 'numeric',
          })}
        </span>
        <button className="btn btn-ghost btn-sm" style={{ position: 'relative' }}>
          <Bell size={16} />
          {openAlerts > 0 && (
            <span
              className="nav-badge"
              style={{ position: 'absolute', top: -4, right: -4, fontSize: 9 }}
            >
              {openAlerts}
            </span>
          )}
        </button>
        <button className="btn btn-ghost btn-sm">
          <Settings size={16} />
        </button>
      </div>
    </header>
  );
}
