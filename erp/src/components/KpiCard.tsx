import { type LucideIcon, TrendingUp, TrendingDown } from 'lucide-react';

interface KpiCardProps {
  title: string;
  value: string;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  icon: LucideIcon;
  color: 'blue' | 'emerald' | 'amber' | 'red' | 'purple' | 'cyan';
}

const colorMap = {
  blue: 'bg-blue-50 text-blue-600',
  emerald: 'bg-emerald-50 text-emerald-600',
  amber: 'bg-amber-50 text-amber-600',
  red: 'bg-red-50 text-red-600',
  purple: 'bg-purple-50 text-purple-600',
  cyan: 'bg-cyan-50 text-cyan-600',
};

export default function KpiCard({ title, value, change, trend, icon: Icon, color }: KpiCardProps) {
  return (
    <div className="kpi-card">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-gray-500">{title}</p>
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${colorMap[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      {change && (
        <div className="flex items-center gap-1">
          {trend === 'up' && <TrendingUp className="w-4 h-4 text-emerald-500" />}
          {trend === 'down' && <TrendingDown className="w-4 h-4 text-red-500" />}
          <span
            className={`text-sm font-medium ${
              trend === 'up' ? 'text-emerald-600' : trend === 'down' ? 'text-red-600' : 'text-gray-500'
            }`}
          >
            {change}
          </span>
          <span className="text-sm text-gray-400">vs last month</span>
        </div>
      )}
    </div>
  );
}
