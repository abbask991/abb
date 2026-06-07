import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import HumanResources from './pages/HumanResources';
import Inventory from './pages/Inventory';
import Sales from './pages/Sales';
import Procurement from './pages/Procurement';
import Accounting from './pages/Accounting';
import Projects from './pages/Projects';
import Reports from './pages/Reports';
import Settings from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/hr" element={<HumanResources />} />
          <Route path="/inventory" element={<Inventory />} />
          <Route path="/sales" element={<Sales />} />
          <Route path="/procurement" element={<Procurement />} />
          <Route path="/accounting" element={<Accounting />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
