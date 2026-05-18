import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Transactions from './pages/Transactions';
import Reconciliation from './pages/Reconciliation';
import Ledger from './pages/Ledger';
import Liquidity from './pages/Liquidity';
import Alerts from './pages/Alerts';
import Reports from './pages/Reports';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/transactions" element={<Transactions />} />
          <Route path="/reconciliation" element={<Reconciliation />} />
          <Route path="/ledger" element={<Ledger />} />
          <Route path="/liquidity" element={<Liquidity />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/reports" element={<Reports />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
