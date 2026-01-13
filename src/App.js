import logo from "./logo.svg";
import "./App.css";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Loginsignup } from "./Components/login/signup/loginsignup";
import { Homepage } from "./Components/home_page/homepage";
import { Generatequestion } from "./Components/generate_question/generatequestion";

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

          {/* homepage */}
          <Route path="/home" element={<Homepage />} />

          {/* generate question page */}
          <Route path="/generate" element={<Generatequestion />}/>

        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;