import React from "react";

import { Routes, Route } from "react-router-dom";

import Container from "@mui/material/Container";

import Header from "./common/Header";
import Footer from "./common/Footer";

import MainDemo from "./pages/MainDemo/MainDemo";
import About from "./pages/About";

function App() {
  return (
    <div className="App">
      <Header />
      <Container
        sx={{
          minHeight: "100vh",
          position: "relative",
        }}
      >
        <Routes>
          <Route path="/" element={<MainDemo />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Container>
      <Footer />
    </div>
  );
}

export default App;
