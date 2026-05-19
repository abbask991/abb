type Variant = 'success' | 'warning' | 'danger' | 'info' | 'gray';

interface StatusBadgeProps {
  status: string;
  variant?: Variant;
}

const autoVariant = (status: string): Variant => {
  const s = status.toLowerCase();
  if (['active', 'paid', 'delivered', 'received', 'posted', 'completed', 'done', 'approved', 'in-stock', 'processed', 'closed-won'].includes(s)) return 'success';
  if (['pending', 'confirmed', 'partial', 'in-progress', 'review', 'on-leave', 'low-stock', 'proposal', 'qualified', 'negotiation', 'sent', 'planning', 'on-hold'].includes(s)) return 'warning';
  if (['cancelled', 'failed', 'overdue', 'out-of-stock', 'terminated', 'blacklisted', 'rejected', 'closed-lost', 'reversed'].includes(s)) return 'danger';
  if (['draft', 'lead', 'todo'].includes(s)) return 'gray';
  return 'info';
};

const variantClasses: Record<Variant, string> = {
  success: 'badge-success',
  warning: 'badge-warning',
  danger: 'badge-danger',
  info: 'badge-info',
  gray: 'badge-gray',
};

export default function StatusBadge({ status, variant }: StatusBadgeProps) {
  const v = variant ?? autoVariant(status);
  const label = status.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  return <span className={variantClasses[v]}>{label}</span>;
}
