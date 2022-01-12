import React from "react";
import Container from "@mui/material/Container";
import { styled } from "@mui/material/styles";

const Div = styled("div")(({ theme }) => ({
  ...theme.typography,
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(1),
}));

function Home() {
  return (
    <Container>
      <Div>Home Placeholder</Div>
      <p>nice this is a home page with some stuff?</p>
      <p>this might be for other demos or explanations or something</p>
      <p>probably will rename from pages to components or demos or something</p>
    </Container>
  );
}

export default Home;
