import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  ArrowLeftRight,
  GitCompareArrows,
  BookOpen,
  Droplets,
  AlertTriangle,
  BarChart3,
} from 'lucide-react';
import { alerts } from '../data/mockData';

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/transactions', label: 'Transactions', icon: ArrowLeftRight },
  { to: '/reconciliation', label: 'Reconciliation', icon: GitCompareArrows },
  { to: '/ledger', label: 'Ledger', icon: BookOpen },
  { to: '/liquidity', label: 'Liquidity', icon: Droplets },
  {
    to: '/alerts',
    label: 'Alerts & Exceptions',
    icon: AlertTriangle,
    badge: alerts.filter((a) => a.status === 'open').length,
  },
  { to: '/reports', label: 'Reports & Analytics', icon: BarChart3 },
];

export default function Sidebar() {
  const location = useLocation();

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>FinanceOps</h1>
        <span>Automation Platform</span>
      </div>
      <nav className="sidebar-nav">
        <div className="nav-section-label">Navigation</div>
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={() =>
              `nav-item${location.pathname === item.to || (item.to !== '/' && location.pathname.startsWith(item.to)) ? ' active' : ''}`
            }
          >
            <item.icon />
            {item.label}
            {item.badge ? <span className="nav-badge">{item.badge}</span> : null}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
