import React from "react";
import styled from "styled-components";
import Container from "@mui/material/Container";
import { Typography } from "@mui/material";

const PageTitle = styled.h2`
  font-family: "Roboto", sans-serif;
  font-size: 1.5rem;
`;

function About() {
  return (
    <Container>
      <PageTitle>About</PageTitle>
      <Typography
        sx={{
          fontFamily: "Roboto",
          fontSize: "1.5rem",
          m: 40,
        }}
      >
        Placeholder Content
      </Typography>
    </Container>
  );
}

export default About;
