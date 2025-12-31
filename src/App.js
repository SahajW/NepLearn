import logo from "./logo.svg";
import "./App.css";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Loginsignup } from "./Components/login/signup/loginsignup";
import { Homepage } from "./Components/home_page/homepage";

function App() {
  return (
    <div>
      {/*<Loginsignup />
      <Homepage/>*/}
      <BrowserRouter>
        <Routes>
          {/* 2. Define your paths */}

          {/* This is the default page (Login) */}
          <Route path="/" element={<Loginsignup />} />

          {/* This is the new Homepage */}
          <Route path="/home" element={<Homepage />} />

        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;