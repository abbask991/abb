import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Users, Package, ShoppingCart, Truck,
  BookOpen, FolderKanban, BarChart3, Settings, ChevronLeft,
  ChevronRight, Building2,
} from 'lucide-react';

const navItems = [
  { label: 'Dashboard', path: '/', icon: LayoutDashboard },
  { label: 'Human Resources', path: '/hr', icon: Users },
  { label: 'Inventory', path: '/inventory', icon: Package },
  { label: 'Sales', path: '/sales', icon: ShoppingCart },
  { label: 'Procurement', path: '/procurement', icon: Truck },
  { label: 'Accounting', path: '/accounting', icon: BookOpen },
  { label: 'Projects', path: '/projects', icon: FolderKanban },
  { label: 'Reports', path: '/reports', icon: BarChart3 },
  { label: 'Settings', path: '/settings', icon: Settings },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <aside
      className={`fixed left-0 top-0 h-screen bg-gray-900 text-white flex flex-col transition-all duration-300 z-30 ${
        collapsed ? 'w-16' : 'w-60'
      }`}
    >
      <div className="flex items-center gap-3 px-4 h-16 border-b border-gray-800">
        <Building2 className="w-8 h-8 text-blue-400 shrink-0" />
        {!collapsed && (
          <span className="text-lg font-bold tracking-tight whitespace-nowrap">
            Enterprise ERP
          </span>
        )}
      </div>

      <nav className="flex-1 overflow-y-auto py-4 scrollbar-thin">
        <ul className="space-y-1 px-2">
          {navItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800'
                  }`
                }
                title={collapsed ? item.label : undefined}
              >
                <item.icon className="w-5 h-5 shrink-0" />
                {!collapsed && <span>{item.label}</span>}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      <button
        onClick={onToggle}
        className="flex items-center justify-center h-12 border-t border-gray-800 text-gray-400 hover:text-white transition-colors cursor-pointer"
      >
        {collapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
      </button>
    </aside>
  );
}
