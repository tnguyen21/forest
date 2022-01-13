import React from "react";
import styled from "styled-components";
import Container from "@mui/material/Container";

const PageTitle = styled.h2`
  font-family: "Roboto", sans-serif;
  font-size: 1.5rem;
`;

const Content = styled.div`
  padding: 500px 0;
`;

function About() {
  return (
    <Container>
      <PageTitle>About</PageTitle>
      <Content>content</Content>
    </Container>
  );
}

export default About;
