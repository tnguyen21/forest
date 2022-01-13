import React from "react";
import styled from "styled-components";

import GitHubIcon from "@mui/icons-material/GitHub";

const FooterContainer = styled.footer`
  padding: 0 8rem;
`;

const Credits = styled.div`
  display: inline-block;
  margin: auto 0.5rem;
  font-size: 0.8rem;
  font-family: "Roboto mono", monospace;
`;

const CreditLink = styled.a`
  text-decoration: none;
  color: rgb(50, 87, 209);
  font-weight: bold;

  &:visited {
    color: rgb(50, 87, 209);
  }
`;

const SocialIcon = styled.a`
  float: right;
  color: black;

  &:visited {
    color: black;
  }
`;

function Footer() {
  return (
    <FooterContainer>
      <Credits>
        Developed by{" "}
        <CreditLink href="http://github.com/tnguyen21">
          Tommy Bui Nguyen
        </CreditLink>{" "}
        for Senior Design at{" "}
        <CreditLink href="https://drexel.edu/cci/">
          Drexel University
        </CreditLink>
      </Credits>
      <SocialIcon href="https://github.com/tnguyen21/trie">
        <GitHubIcon />
      </SocialIcon>
    </FooterContainer>
  );
}

export default Footer;
