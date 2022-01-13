import React from "react";

import { Routes, Route } from "react-router-dom";

import Header from "./common/Header";
import Footer from "./common/Footer";

import MainDemo from "./pages/MainDemo/MainDemo";
import About from "./pages/About";

function App() {
  return (
    <div className="App">
      <Header />
      <Routes>
        <Route path="/" element={<MainDemo />} />
        <Route path="/about" element={<About />} />
      </Routes>
      <Footer />
    </div>
  );
}

export default App;
