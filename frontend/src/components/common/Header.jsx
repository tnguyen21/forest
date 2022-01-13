import React from "react";
import styled from "styled-components";

import { Link } from "react-router-dom";

const HeaderContainer = styled.header`
  padding: 0 8rem;
`;

const Title = styled.h1`
  font-family: "Roboto", sans-serif;
  font-size: 1.5rem;
  letter-spacing: 0.15rem;
  margin: 0;
  padding: 0;
  display: inline-block;
`;

const StyledLink = styled(Link)`
  font-family: "Roboto mono", monospace;
  font-size: 1rem;
  text-decoration: none;
  padding: 0.5rem;
  margin: auto 0.5rem;
  float: right;
  &:hover {
    text-decoration: underline;
  }
`;

function Header() {
  return (
    <HeaderContainer>
      <Title>Drexel Senior Design</Title>
      <StyledLink to="/">Home</StyledLink>
      <StyledLink to="/about">About</StyledLink>
    </HeaderContainer>
  );
}

export default Header;
