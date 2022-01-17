import React from "react";
import styled from "styled-components";

import { Link } from "react-router-dom";

const HeaderContainer = styled.header`
  padding: 0 8rem;
  height: 4rem;

  @media screen and (max-width: 600px) {
    & {
      padding: 0;
    }
  }
`;

const Title = styled(Link)`
  font-family: "Roboto mono", sans-serif;
  font-size: 1.5rem;
  font-weight: bold;
  letter-spacing: 0.1rem;
  margin: 0;
  padding: 0;
  display: inline-block;
  text-decoration: none;
  color: black;
`;

const StyledLink = styled(Link)`
  font-family: "Roboto mono", monospace;
  font-size: 1rem;
  text-decoration: none;
  color: rgb(50, 87, 209);
  padding: 0.5rem;
  margin: auto 0.5rem;
  float: right;

  &:hover {
    font-weight: bold;
  }

  &:visited {
    color: rgb(50, 87, 209);
  }
`;

function Header() {
  return (
    <HeaderContainer>
      <Title to="/">NER Demo</Title>
      <StyledLink to="/">Home</StyledLink>
      <StyledLink to="/about">About</StyledLink>
    </HeaderContainer>
  );
}

export default Header;
