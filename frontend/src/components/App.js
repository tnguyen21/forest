import React from "react";

import { Routes, Route } from "react-router-dom";

import Header from "./common/Header";
import Footer from "./common/Footer";

import Home from "./pages/Home";

function App() {
  return (
    <div className="App">
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/test" element={<div>placeholder test</div>} />
      </Routes>
      <Footer />
    </div>
  );
}

export default App;
